"""RF-DETR-Seg（DINOv2系）を 1024 クロップで学習する。

MaskDINO と独立の系統（別バックボーン・別実装・別データパイプライン）なので、
票決に本物の多様性が入る。5-fold アンサンブル（同一構成）が失敗した原因の解消。

VRAM: batch2 + grad_accum4（実効8、MaskDINO と同じ）、勾配チェックポイント有効。
      スモーク実測 batch1 で 6.09GiB。
"""
import torch

torch.cuda.set_per_process_memory_fraction(0.85)   # WSL2 フリーズ対策（必須）
from rfdetr import RFDETRSegPreview

m = RFDETRSegPreview(resolution=1008, num_classes=1, gradient_checkpointing=True)
m.train(dataset_dir="datasets/rfdetr",
        epochs=40, batch_size=2, grad_accum_steps=4,
        output_dir="work_dirs/rfdetr_f0",
        num_workers=4, tensorboard=False, wandb=False,
        early_stopping=True, early_stopping_patience=8,
        checkpoint_interval=5, progress_bar=False)
print("[done] peak", round(torch.cuda.max_memory_allocated() / 2**30, 2), "GiB")
