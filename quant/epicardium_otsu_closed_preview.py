"""
Raw-Otsu tissue mask (see epicardium_otsu_preview.py's --raw path) with
morphological closing applied, to fill the speckled holes/gaps that show up
in the plain mask.

Closing = dilate then erode (cv2.MORPH_CLOSE). Dilating first bridges small
gaps and fills small holes; eroding back afterward shrinks the mask back
toward its original boundary without reopening what got filled. Net effect:
small speckled holes inside tissue get filled in, thin breaks in the mask
get bridged, and the overall tissue outline is left close to unchanged
(closing does not touch large background regions or shrink large real
holes -- only ones small relative to the kernel).

Reuses the full-res TIFF loading and raw-Otsu tissue mask from
epicardium_otsu_preview.py (the CASES path table, the vips-based
downsampler, and otsu_tissue(bgr, raw=True)) rather than re-deriving them.

Writes one 3-panel PNG per case (raw red channel | raw-otsu tissue | closed)
to epicardium_previews_closed/ in the project root. Does not touch
epicardium_previews/ (the un-closed raw-Otsu previews) or any other preview
folder from earlier diagnostic scripts.

Usage:
    python quant/epicardium_otsu_closed_preview.py
    python quant/epicardium_otsu_closed_preview.py --close-radius 5
"""
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

from epicardium_otsu_preview import (CASES, RGB_ROOT, DEFAULT_MAX_DIM,
                                     load_rgb_downsampled, otsu_tissue,
                                     tint, label_panel, TISSUE_COLOR)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / 'epicardium_previews_closed'
DEFAULT_CLOSE_RADIUS = 5  # px, at the 2000px preview resolution


def close_mask(mask, close_radius):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1,) * 2)
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k)
    return closed.astype(bool)


def process_case(stem, rel_path, outdir, max_dim, close_radius, tmpdir):
    src = RGB_ROOT / rel_path
    if not src.exists():
        return f'{stem}: NOT FOUND ({src}) -- skipped'

    case_tmp = tmpdir / stem
    case_tmp.mkdir(parents=True, exist_ok=True)
    bgr, native_hw = load_rgb_downsampled(src, max_dim, case_tmp)
    for f in case_tmp.iterdir():
        f.unlink()
    case_tmp.rmdir()

    tissue = otsu_tissue(bgr, raw=True)
    closed = close_mask(tissue, close_radius)

    p1 = label_panel(bgr, f'{stem}  |  native {native_hw[1]}x{native_hw[0]}')
    p2 = label_panel(tint(bgr, tissue, TISSUE_COLOR),
                     f'raw otsu  {100 * tissue.mean():.1f}% of frame')
    p3 = label_panel(tint(bgr, closed, TISSUE_COLOR),
                     f'closed (r={close_radius})  {100 * closed.mean():.1f}% of frame')

    out_path = outdir / f'{stem}.png'
    cv2.imwrite(str(out_path), np.hstack([p1, p2, p3]))
    return (f'{stem}: native {native_hw[1]}x{native_hw[0]}, '
            f'raw {100*tissue.mean():.1f}% -> closed {100*closed.mean():.1f}%')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument('--max-dim', type=int, default=DEFAULT_MAX_DIM)
    ap.add_argument('--close-radius', type=int, default=DEFAULT_CLOSE_RADIUS)
    args = ap.parse_args()

    if args.outdir.exists():
        for f in args.outdir.glob('*.png'):
            f.unlink()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='epicardium_closed_vips_') as tmp:
        tmpdir = Path(tmp)
        for stem, rel_path in CASES.items():
            print(process_case(stem, rel_path, args.outdir, args.max_dim, args.close_radius, tmpdir),
                  flush=True)
    print(f'\nwrote previews to {args.outdir}')


if __name__ == '__main__':
    main()
