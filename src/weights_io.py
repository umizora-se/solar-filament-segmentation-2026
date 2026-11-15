"""Restore the split weight files in release/weights (see scripts/export_weights.py)."""
import hashlib
import json
from pathlib import Path


def assemble(weights_dir, name):
    """Join `<name>.part00, .part01, ...` into `<name>` (once) and verify its SHA-256 against MANIFEST.json."""
    d = Path(weights_dir); out = d / name
    if not out.exists():
        parts = sorted(d.glob(f"{name}.part*"))
        assert parts, f"no parts for {name} in {d}"
        with open(out, "wb") as f:
            for p in parts:
                f.write(p.read_bytes())
    want = json.loads((d / "MANIFEST.json").read_text())[name]["sha256"]
    got = hashlib.sha256(out.read_bytes()).hexdigest()
    assert got == want, f"{name}: checksum mismatch"
    return out
