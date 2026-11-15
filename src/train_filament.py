"""
MaskDINO フィラメント・ベースライン学習エントリ。

例:
  # fold 0 を学習（COCO 事前学習を自動DL）
  python src/train_filament.py --config configs/maskdino_filament_r50_1024crop.yaml --fold 0

  # スモークテスト（20 iter だけ）
  python src/train_filament.py --config configs/maskdino_filament_r50_1024crop.yaml \
      --fold 0 --smoke

  # 学習済みで val PQ 評価のみ
  python src/train_filament.py --config configs/maskdino_filament_r50_1024crop.yaml \
      --fold 0 --eval-only MODEL.WEIGHTS work_dirs/r50_f0/model_final.pth
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "external" / "MaskDINO"))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import DatasetMapper
from detectron2.engine import AMPTrainer, default_setup, launch
from detectron2.utils import comm

import train_net as maskdino_train_net  # external/MaskDINO/train_net.py
from filament_config import setup_cfg
from filament_dataset import register_filament
from filament_mapper import FilamentInstanceDatasetMapper
from pq_evaluator import FilamentPQEvaluator

PRETRAINED = {
    # 3-scale, hid1024 -> configs/.../maskdino_R50_bs16_50ep_3s.yaml と一致
    "r50_3s": ("https://github.com/IDEA-Research/detrex-storage/releases/download/"
               "maskdino-v0.1.0/maskdino_r50_50ep_300q_hid1024_3sd1_instance_"
               "maskenhanced_mask46.1ap_box51.5ap.pth"),
    # Swin-L, 4-scale(input_proj 5レベル) -> configs/.../swin/..._4s_dowsample1_2048.yaml
    # COCO mask 52.3 AP（R50 は 46.1 AP）
    "swinl_4s": ("https://github.com/IDEA-Research/detrex-storage/releases/download/"
                 "maskdino-v0.1.0/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_"
                 "maskenhanced_mask52.3ap_box59.0ap.pth"),
}


def maybe_download_pretrained(key: str = "r50_3s") -> str:
    dst = REPO / "work_dirs" / "pretrained" / Path(PRETRAINED[key]).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        import urllib.request
        print(f"[pretrained] downloading {PRETRAINED[key]}")
        urllib.request.urlretrieve(PRETRAINED[key], dst)
    print(f"[pretrained] {dst}  ({dst.stat().st_size/1e6:.1f} MB)")
    return str(dst)


def acquire_run_lock(output_dir: str, force: bool = False):
    """同じ OUTPUT_DIR への二重起動を防ぐ。

    二重起動すると GPU を奪い合って WSL がフリーズし、さらに同じ
    チェックポイント/ログに2プロセスが書いて破損する恐れがある。
    """
    import atexit, json as _json
    lock = Path(output_dir) / "train.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)

    if lock.exists() and not force:
        try:
            old_pid = int(_json.loads(lock.read_text())["pid"])
        except Exception:
            old_pid = None
        alive = False
        if old_pid:
            try:
                cmd = Path(f"/proc/{old_pid}/cmdline").read_bytes().decode("utf-8", "ignore")
                alive = "train_filament" in cmd
            except Exception:
                alive = False
        if alive:
            print(f"\n★ 既に学習が実行中です (pid={old_pid}, {output_dir})")
            print("  二重起動は GPU の奪い合いとチェックポイント破損を招くため中止します。")
            print(f"  止めるなら:  kill {old_pid}")
            print("  それでも起動するなら --force を付けてください。\n")
            sys.exit(1)
        print(f"[lock] 古いロック(pid={old_pid})を掃除しました")

    owner = os.getpid()
    lock.write_text(_json.dumps({"pid": owner, "output_dir": output_dir}))
    atexit.register(lambda: lock.unlink(missing_ok=True) if os.getpid() == owner else None)
    print(f"[lock] {lock} を取得 (pid={owner})")


def apply_vram_guards(cfg):
    """VRAM 16GB で安全に回すためのガード。学習・推論の両方で必ず最初に呼ぶ。"""
    # 1) メモリ上限: WSL2 は GPU メモリ溢れをホストRAMへページングしてシステムを
    #    フリーズさせる。上限を切ることで clean な CUDA OutOfMemoryError にする。
    if torch.cuda.is_available():
        frac = cfg.FILAMENT.CUDA_MEM_FRACTION
        torch.cuda.set_per_process_memory_fraction(frac)
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        print(f"[vram-guard] {torch.cuda.get_device_name(0)}  "
              f"total={total:.1f} GiB  上限={frac:.0%} ({total*frac:.1f} GiB)")
    # 2) 勾配チェックポイント: deformable attention エンコーダの活性化を保存せず
    #    backward 時に再計算する（1024crop bs2 で 12.4 -> 7.0 GiB）。
    os.environ["FILAMENT_GRAD_CKPT"] = "1" if cfg.FILAMENT.GRAD_CHECKPOINT else "0"
    print(f"[vram-guard] grad_checkpoint={cfg.FILAMENT.GRAD_CHECKPOINT}  "
          f"accum_steps={cfg.SOLVER.ACCUM_STEPS}  "
          f"ims_per_batch={cfg.SOLVER.IMS_PER_BATCH} "
          f"-> 実効バッチ {cfg.SOLVER.IMS_PER_BATCH * cfg.SOLVER.ACCUM_STEPS}")
    # 3) マスク損失の切替（criterion.py が環境変数で分岐）
    os.environ["FILAMENT_TVERSKY"] = "1" if cfg.FILAMENT.TVERSKY_LOSS else "0"
    os.environ["FILAMENT_HIRES_MASK"] = "1" if cfg.FILAMENT.HIRES_MASK else "0"
    os.environ["FILAMENT_HIRES_SKIP"] = "1" if cfg.FILAMENT.HIRES_SKIP else "0"
    os.environ["FILAMENT_POINT_MASK"] = "1" if cfg.FILAMENT.POINT_MASK else "0"
    os.environ["FILAMENT_IOU_CLS"] = "1" if cfg.FILAMENT.IOU_CLS else "0"
    print(f"[loss] cls_target = {'mask IoU (IoU-aware)' if cfg.FILAMENT.IOU_CLS else 'binary'}")
    print(f"[loss] mask_loss = {'Focal-Tversky' if cfg.FILAMENT.TVERSKY_LOSS else 'Dice'}")
    # 4) 断片化対策
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class AccumAMPTrainer(AMPTrainer):
    """勾配累積つき AMPTrainer。VRAM を増やさずに実効バッチを稼ぐ。

    accum_steps 回ぶん forward/backward してから 1 回だけ optimizer.step() する。
    損失は 1/accum_steps でスケールするので勾配は平均になる。
    """

    accum_steps = 1

    def run_step(self):
        assert self.model.training, "AccumAMPTrainer は train モードでのみ使う"
        from torch.amp import autocast

        self.optimizer.zero_grad(set_to_none=True)
        total_data_time = 0.0
        accum_losses = {}

        for _ in range(self.accum_steps):
            t0 = time.perf_counter()
            data = next(self._data_loader_iter)
            total_data_time += time.perf_counter() - t0

            with autocast("cuda", dtype=self.precision):
                loss_dict = self.model(data)
                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                else:
                    losses = sum(loss_dict.values())

            self.grad_scaler.scale(losses / self.accum_steps).backward()
            for k, v in loss_dict.items():
                accum_losses[k] = accum_losses.get(k, 0.0) + v.detach() / self.accum_steps

        # full_model クリッピングは optimizer.step() 内で行われる。
        # grad_scaler.step() が先に unscale_ するのでスケール済み勾配は渡らない。
        self._write_metrics(accum_losses, total_data_time)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()


class FilamentTrainer(maskdino_train_net.Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        accum = cfg.SOLVER.ACCUM_STEPS
        if accum > 1:
            assert cfg.SOLVER.AMP.ENABLED, "ACCUM_STEPS>1 は SOLVER.AMP.ENABLED=True が前提"
            # 既存の AMPTrainer を差し替える（model/data_loader/optimizer は流用）
            self._trainer.__class__ = AccumAMPTrainer
            self._trainer.accum_steps = accum

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME == "filament_instance":
            mapper = FilamentInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        return super().build_train_loader(cfg)

    def build_hooks(self):
        hooks_list = super().build_hooks()
        # val-PQ が最良の重みを別途保存する。長時間学習では model_final が
        # 最良とは限らないため（過学習で後半に劣化しうる）。
        if self.cfg.TEST.EVAL_PERIOD > 0:
            from detectron2.engine import hooks as d2hooks
            hooks_list.append(d2hooks.BestCheckpointer(
                self.cfg.TEST.EVAL_PERIOD, self.checkpointer,
                "filament_pq/PQ", mode="max", file_prefix="model_best"))
        if not self.cfg.FILAMENT.EVAL_AFTER_TRAIN:
            # detectron2 の EvalHook は EVAL_PERIOD=0 でも after_train で必ず評価する。
            # スモーク時はこれを止める（val 142枚 = 約11分の無駄を避ける）。
            for h in hooks_list:
                if hasattr(h, "_eval_after_train"):
                    h._eval_after_train = False
        return hooks_list

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # ネイティブ 2048 のまま（MIN_SIZE_TEST=0 でリサイズ無し、2048 は /32 なのでパディング無し）
        mapper = DatasetMapper(is_train=False, augmentations=[],
                               image_format=cfg.INPUT.FORMAT)
        n = cfg.FILAMENT.EVAL_MAX_IMAGES
        if n and n > 0:
            # 学習中の評価を間引く（フル2048推論は 4.5 s/枚）。決定的にサンプリング。
            from detectron2.data import DatasetCatalog
            dicts = DatasetCatalog.get(dataset_name)
            if len(dicts) > n:
                step = len(dicts) / n
                dicts = [dicts[int(i * step)] for i in range(n)]
                print(f"[eval] {dataset_name} を {n} 枚に間引いて評価します"
                      f"（最終値は --eval-only + FILAMENT.EVAL_MAX_IMAGES 0 で取得）")
            return build_detection_test_loader(dicts, mapper=mapper)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return FilamentPQEvaluator(dataset_name, cfg, output_folder)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="20 iter だけ回して健全性確認")
    ap.add_argument("--output-dir", default=None,
                    help="出力先。未指定なら config の OUTPUT_DIR を使う")
    ap.add_argument("--force", action="store_true",
                    help="二重起動ガードを無視して起動する")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="COCO 事前学習を使わず ImageNet R50 から")
    ap.add_argument("opts", nargs=argparse.REMAINDER,
                    help="cfg 上書き (KEY VALUE ...)")
    return ap


def main(args):
    args.config_file = args.config  # default_setup / detectron2 互換
    opts = list(args.opts)
    if opts and opts[0] == "--":
        opts = opts[1:]

    fold = args.fold
    # OUTPUT_DIR は「明示指定 > opts の OUTPUT_DIR > config ファイルの値」の順で決める。
    # 以前はここで work_dirs/r50_f{fold} を無条件に上書きしており、
    # 別 config / 別 OUTPUT_DIR を指定しても既存の学習結果を壊していた。
    out_dir = args.output_dir            # None なら setup_cfg 側で config の値を使う
    opts += ["DATASETS.TRAIN", f"('filament_train_f{fold}',)",
             "DATASETS.TEST", f"('filament_val_f{fold}',)"]

    if args.smoke:
        opts += ["SOLVER.MAX_ITER", "20", "SOLVER.STEPS", "()",
                 "SOLVER.WARMUP_ITERS", "5", "TEST.EVAL_PERIOD", "0",
                 "SOLVER.CHECKPOINT_PERIOD", "999999", "DATALOADER.NUM_WORKERS", "2",
                 "FILAMENT.DUMP_PRED", "False",
                 "FILAMENT.EVAL_AFTER_TRAIN", "False"]   # 最終評価(約11分)を省く
        out_dir = args.output_dir or str(REPO / "work_dirs" / "smoke")

    cfg = setup_cfg(args.config, opts, output_dir=out_dir)
    if cfg.FILAMENT.USE_NEIGHBORS:
        # 近傍 GONG フレームへのラベル転写（Kaggle discussion 748045 で許可）。
        # 学習データだけを差し替え、val は元画像のまま（filament_dataset 側で
        # 近傍が元画像と同じ fold に所属することを保証している）。
        cfg.defrost()
        cfg.DATASETS.TRAIN = (f"filament_train_nb_f{fold}",)
        cfg.freeze()
        print(f"[neighbors] DATASETS.TRAIN = {cfg.DATASETS.TRAIN}", flush=True)
    if cfg.FILAMENT.USE_PSEUDO:
        # 外部 GONG フレーム（学習・test のある日を除外）に同じ fold の教師で付けた疑似ラベル
        assert not cfg.FILAMENT.USE_NEIGHBORS
        cfg.defrost()
        cfg.DATASETS.TRAIN = (f"filament_train_pl_f{fold}",)
        cfg.freeze()
        print(f"[pseudo] DATASETS.TRAIN = {cfg.DATASETS.TRAIN}", flush=True)

    # 事前学習の重み
    if not args.eval_only and not args.resume:
        if args.no_pretrained:
            pass  # yaml の ImageNet R50 のまま
        else:
            w = maybe_download_pretrained(cfg.FILAMENT.PRETRAINED_KEY)
            cfg.defrost(); cfg.MODEL.WEIGHTS = w; cfg.freeze()

    if not args.eval_only:
        acquire_run_lock(cfg.OUTPUT_DIR, force=args.force)
    apply_vram_guards(cfg)
    default_setup(cfg, args)
    register_filament(fold)

    if args.eval_only:
        model = FilamentTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume)
        return FilamentTrainer.test(cfg, model)

    trainer = FilamentTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = build_argparser().parse_args()
    print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
    launch(main, args.num_gpus, dist_url="auto", args=(args,))
