"""GONG Hα FITS を、競技画像と同一の JPEG に変換する。

手順は主催者公開の MLEco FITS2JPEG Convertor（bitbucket.org/dataresearchlab/
mleco-fits2jpegconvertor）と同じ:
    funpack → wavelet（縁強調フィルタ）→ 灰色化 → JPEG
灰色化は同ツールの fits2ppmColorscale.cc（grey, min=640, max=5335）を numpy で
再実装した。同ツールは 52MB のテキスト形式 PPM を経由するため1枚に数秒かかるが、
ここでは直接 JPEG を書く。

    灰色化: v を [640, 5335] に切り詰め → rint(255·(v-640)/(5335-640)) → 上下反転
    JPEG  : 単一チャンネル、品質92（ImageMagick convert の既定値）

検証: 学習画像 20140609195854Bh について、競技提供の JPEG と画素が完全一致
（平均誤差 0.000、最大誤差 0）。funpack と wavelet は cfitsio 4.7.0 で自前ビルド。
"""
from __future__ import annotations

import io
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
WAVELET = REPO / "tools" / "fits2jpeg" / "wavelet"
FUNPACK = Path.home() / "local" / "bin" / "funpack"
VMIN, VMAX = 640.0, 5335.0
JPEG_QUALITY = 92
_ENV = dict(os.environ, LD_LIBRARY_PATH=str(Path.home() / "local" / "lib"))


def grey_from_wavelet(data: np.ndarray) -> np.ndarray:
    """fits2ppmColorscale.cc の grey 写像と同一の結果を返す。"""
    v = np.clip(data.astype(np.float64), VMIN, VMAX)
    idx = np.rint(255.0 * (v - VMIN) / (VMAX - VMIN)).astype(np.uint8)
    return idx[::-1]                      # FITS と画像で Y 軸が逆


def fz_to_jpeg(fz_path: str | Path, out_jpeg: str | Path) -> None:
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw.fits"
        wav = Path(td) / "wav.fits"
        subprocess.run([str(FUNPACK), "-O", str(raw), str(fz_path)],
                       check=True, env=_ENV, capture_output=True)
        subprocess.run([str(WAVELET), str(raw), str(wav)],
                       check=True, env=_ENV, capture_output=True)
        data = fits.getdata(wav)
    img = grey_from_wavelet(data)
    Path(out_jpeg).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, "L").save(out_jpeg, "JPEG", quality=JPEG_QUALITY)
