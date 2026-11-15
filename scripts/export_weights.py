"""Export the 10 inference weights (5 folds x {seed-1 SWA, seed 2}) as model-only files split into <100 MB parts.

    python3 scripts/export_weights.py --out release/weights
GitHub rejects files above 100 MB, so each .pth is cut into 90 MB parts (`*.pth.part00`, ...);
`src/weights_io.py::assemble` (or `cat parts > file`) restores a byte-identical file.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
PART = 90_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "release/weights"))
    out = Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for k in range(5):
        srcs = {f"fold{k}_seed1_swa": REPO / f"work_dirs/r50_full2048_f{k}/model_swaA.pth",
                f"fold{k}_seed2": REPO / f"work_dirs/r50_full2048_seed2_f{k}/model_final.pth"}
        for name, p in srcs.items():
            ck = torch.load(p, map_location="cpu", weights_only=False)
            f = out / f"{name}.pth"
            torch.save({"model": ck["model"], "iteration": ck.get("iteration", -1)}, f)
            data = f.read_bytes()
            manifest[f.name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for i in range(0, len(data), PART):
                (out / f"{f.name}.part{i // PART:02d}").write_bytes(data[i:i + PART])
            f.unlink()
            print(name, len(data) // 10**6, "MB", flush=True)
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
