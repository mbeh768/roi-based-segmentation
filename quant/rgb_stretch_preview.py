"""
Plain diagnostic preview: full-color RGB, linearly contrast-stretched to use
the entire 8-bit range, after excluding completely black pixels (R=G=B=0 --
scanner/stitching artifacts, gaps between scan tiles, unscanned borders; see
epicardium_otsu_preview.py's docstring for why these aren't background and
shouldn't be treated as part of the real intensity distribution).

Per channel, min and max are taken over only the non-artifact ("real")
pixels, and that channel is linearly rescaled so its min maps to 0 and its
max maps to 255. Artifact pixels stay at 0 in the output. Purely visual --
like cardiomyocyte_gamma_preview.py, this is for inspecting the tissue/noise
distribution by eye, not for feeding into a thresholding step.

Reuses the full-res TIFF loading from epicardium_otsu_preview.py (the CASES
path table and the vips-based downsampler that handles this dataset's three
on-disk layouts) rather than re-deriving it.

Writes one 2-panel PNG per case (original | contrast-stretched) to
rgb_stretch_previews/ in the project root. Does not touch epicardium_previews/
or cardiomyocyte_gamma_previews/ from the other diagnostic scripts.

Usage:
    python quant/rgb_stretch_preview.py
    python quant/rgb_stretch_preview.py --max-dim 2000
"""
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

from epicardium_otsu_preview import CASES, RGB_ROOT, DEFAULT_MAX_DIM, load_rgb_downsampled

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / 'rgb_stretch_previews'


def stretch_per_channel(bgr, real):
    out = np.zeros_like(bgr)
    for c in range(3):
        chan = bgr[..., c]
        vals = chan[real]
        if vals.size == 0:
            continue
        lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            out[..., c] = np.where(real, chan, 0)
            continue
        stretched = (chan.astype(np.float32) - lo) / (hi - lo) * 255.0
        out[..., c] = np.clip(stretched, 0, 255).astype(np.uint8)
    out[~real] = 0
    return out


def label_panel(bgr, text):
    bgr = bgr.copy()
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(bgr, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def process_case(stem, rel_path, outdir, max_dim, tmpdir):
    src = RGB_ROOT / rel_path
    if not src.exists():
        return f'{stem}: NOT FOUND ({src}) -- skipped'

    case_tmp = tmpdir / stem
    case_tmp.mkdir(parents=True, exist_ok=True)
    bgr, native_hw = load_rgb_downsampled(src, max_dim, case_tmp)
    for f in case_tmp.iterdir():
        f.unlink()
    case_tmp.rmdir()

    real = np.any(bgr > 0, axis=-1)
    stretched = stretch_per_channel(bgr, real)

    p1 = label_panel(bgr, f'{stem}  |  native {native_hw[1]}x{native_hw[0]}  '
                          f'-> preview {bgr.shape[1]}x{bgr.shape[0]}  |  original')
    p2 = label_panel(stretched, 'contrast-stretched (per channel, 0-pixels excluded)')

    out_path = outdir / f'{stem}.png'
    cv2.imwrite(str(out_path), np.hstack([p1, p2]))
    return f'{stem}: native {native_hw[1]}x{native_hw[0]}, wrote {out_path.name}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument('--max-dim', type=int, default=DEFAULT_MAX_DIM)
    args = ap.parse_args()

    if args.outdir.exists():
        for f in args.outdir.glob('*.png'):
            f.unlink()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='rgb_stretch_vips_') as tmp:
        tmpdir = Path(tmp)
        for stem, rel_path in CASES.items():
            print(process_case(stem, rel_path, args.outdir, args.max_dim, tmpdir), flush=True)
    print(f'\nwrote previews to {args.outdir}')


if __name__ == '__main__':
    main()
