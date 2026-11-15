"""TTA vote score: score' = score * (votes / n_views) ** alpha.

Input is a raw prediction JSON from `predict_submit.py --dump-raw` (each item has rle, score, votes, nv).
The number of TTA views that agree on an instance is a confidence cue independent of the model score.

    python src/vote_rescore.py --preds test_raw.json --alpha 0.75 --out test_raw_a0.75.json
"""
import argparse
import json
from pathlib import Path


def rescore(preds: dict, alpha: float) -> dict:
    return {s: [{"rle": it["rle"], "score": it["score"] * (it["votes"] / it["nv"]) ** alpha} for it in items]
            for s, items in preds.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = json.loads(Path(a.preds).read_text())["preds"]
    Path(a.out).write_text(json.dumps({"preds": rescore(raw, a.alpha)}))
    print(f"[vote_rescore] {len(raw)} images, alpha={a.alpha} -> {a.out}")


if __name__ == "__main__":
    main()
