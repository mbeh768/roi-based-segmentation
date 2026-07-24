"""
Plain diagnostic preview of the cardiomyocyte (red) channel alone, gamma-
corrected, for visually inspecting the noise/tissue intensity distribution
before picking a thresholding method -- no Otsu, no tissue mask, no
statistics, just the channel itself.

Gamma 0.4 (output = (input/255)^0.4 * 255) pulls the low end of the range up
disproportionately: dim real tissue and faint background noise, which sit
close together near 0 in the raw channel and are hard to tell apart by eye,
get spread out over a much wider visible range. Bright pixels are compressed
less. This is purely visual -- the saved PNG is only ever meant to be looked
at, not fed back into a thresholding step.

Reuses the full-res TIFF loading from epicardium_otsu_preview.py (the
CASES path table and the vips-based downsampler that handles this dataset's
three on-disk layouts) rather than re-deriving it.

Writes one 2-panel PNG per case (raw red channel | gamma-corrected) to
cardiomyocyte_gamma_previews/ in the project root.

Usage:
    python quant/cardiomyocyte_gamma_preview.py
    python quant/cardiomyocyte_gamma_preview.py --gamma 0.4 --max-dim 2000
"""
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

from epicardium_otsu_preview import (CASES, RGB_ROOT, DEFAULT_MAX_DIM,
                                     load_rgb_downsampled)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / 'cardiomyocyte_gamma_previews'
DEFAULT_GAMMA = 0.4


def apply_gamma(gray_u8, gamma):
    lut = (np.linspace(0, 1, 256) ** gamma * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.LUT(gray_u8, lut)


def label_panel(gray_or_bgr, text):
    bgr = cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2BGR) if gray_or_bgr.ndim == 2 else gray_or_bgr.copy()
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(bgr, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def process_case(stem, rel_path, outdir, max_dim, gamma, tmpdir):
    src = RGB_ROOT / rel_path
    if not src.exists():
        return f'{stem}: NOT FOUND ({src}) -- skipped'

    case_tmp = tmpdir / stem
    case_tmp.mkdir(parents=True, exist_ok=True)
    bgr, native_hw = load_rgb_downsampled(src, max_dim, case_tmp)
    for f in case_tmp.iterdir():
        f.unlink()
    case_tmp.rmdir()

    red = bgr[..., 2]
    gamma_corrected = apply_gamma(red, gamma)

    p1 = label_panel(red, f'{stem}  |  native {native_hw[1]}x{native_hw[0]}  '
                          f'-> preview {red.shape[1]}x{red.shape[0]}  |  raw red channel')
    p2 = label_panel(gamma_corrected, f'gamma {gamma} (brightened)')

    out_path = outdir / f'{stem}.png'
    cv2.imwrite(str(out_path), np.hstack([p1, p2]))
    return f'{stem}: native {native_hw[1]}x{native_hw[0]}, wrote {out_path.name}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument('--max-dim', type=int, default=DEFAULT_MAX_DIM)
    ap.add_argument('--gamma', type=float, default=DEFAULT_GAMMA)
    args = ap.parse_args()

    if args.outdir.exists():
        for f in args.outdir.glob('*.png'):
            f.unlink()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='cmyocyte_gamma_vips_') as tmp:
        tmpdir = Path(tmp)
        for stem, rel_path in CASES.items():
            print(process_case(stem, rel_path, args.outdir, args.max_dim, args.gamma, tmpdir),
                  flush=True)
    print(f'\nwrote previews to {args.outdir}')


if __name__ == '__main__':
    main()
