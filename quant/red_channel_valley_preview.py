"""
Denoise the red channel (cardiomyocyte stain) by finding the histogram valley
between the low-level noise peak and the tissue signal, then deleting
(zeroing) every pixel at or below it. Values above the valley are left
completely untouched -- no capping, no rescaling, just a floor.

The valley is found the same way you'd read it off red_channel_histogram.py's
plots by eye: smooth the histogram of intensities 0-50 with a +/-2 moving
average (matches that script's log-scale plots, where the noise peak and the
valley leading away from it both sit in that range across this dataset), then
take the position of the minimum smoothed count. A plain argmin is used
rather than walking backward from 50 looking for a derivative sign flip --
tried that first, but on real (noisy) histograms the smoothed derivative
doesn't reliably cross zero at the valley, it just slows down, so a hunt for
an exact zero-crossing is the wrong tool. The minimum position is the same
answer without that fragility.

Reuses the full-res TIFF loading from epicardium_otsu_preview.py (the CASES
path table and the vips-based downsampler that handles this dataset's three
on-disk layouts) rather than re-deriving it.

Writes one 2-panel PNG per case (original red channel | valley-floored) to
red_channel_valley_previews/ in the project root. Does not touch
epicardium_previews/, cardiomyocyte_gamma_previews/, or
red_channel_histograms/ from the other diagnostic scripts.

Usage:
    python quant/red_channel_valley_preview.py
    python quant/red_channel_valley_preview.py --search-max 50 --smooth-radius 2
"""
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

from epicardium_otsu_preview import CASES, RGB_ROOT, DEFAULT_MAX_DIM, load_rgb_downsampled

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / 'red_channel_valley_previews'
DEFAULT_SEARCH_MAX = 50
DEFAULT_SMOOTH_RADIUS = 2


def find_valley(counts, search_max, smooth_radius):
    """Position (intensity 0-255) of the histogram valley between the noise
    peak and the tissue signal, searched within (0, search_max]."""
    w = 2 * smooth_radius + 1
    kernel = np.ones(w) / w
    smoothed = np.convolve(counts.astype(np.float64), kernel, mode='same')
    window = smoothed[1:search_max + 1]
    return 1 + int(np.argmin(window))


def floor_below_valley(red, real, search_max, smooth_radius):
    """Zero every pixel at or below the valley threshold; everything above it
    (and all artifact pixels) is left/set exactly as before. Returns the
    floored channel and the threshold used."""
    vals = red[real]
    counts = np.bincount(vals, minlength=256)[:256]
    threshold = find_valley(counts, search_max, smooth_radius)
    out = red.copy()
    out[out <= threshold] = 0
    out[~real] = 0
    return out, threshold


def label_panel(gray, text):
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(bgr, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def process_case(stem, rel_path, outdir, max_dim, search_max, smooth_radius, tmpdir):
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
    real = np.any(bgr > 0, axis=-1)
    cleaned, threshold = floor_below_valley(red, real, search_max, smooth_radius)

    p1 = label_panel(red, f'{stem}  |  native {native_hw[1]}x{native_hw[0]}  |  raw red channel')
    p2 = label_panel(cleaned, f'floored at valley (threshold={threshold})')

    out_path = outdir / f'{stem}.png'
    cv2.imwrite(str(out_path), np.hstack([p1, p2]))
    return f'{stem}: native {native_hw[1]}x{native_hw[0]}, valley threshold={threshold}, wrote {out_path.name}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument('--max-dim', type=int, default=DEFAULT_MAX_DIM)
    ap.add_argument('--search-max', type=int, default=DEFAULT_SEARCH_MAX)
    ap.add_argument('--smooth-radius', type=int, default=DEFAULT_SMOOTH_RADIUS)
    args = ap.parse_args()

    if args.outdir.exists():
        for f in args.outdir.glob('*.png'):
            f.unlink()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='red_valley_vips_') as tmp:
        tmpdir = Path(tmp)
        for stem, rel_path in CASES.items():
            print(process_case(stem, rel_path, args.outdir, args.max_dim,
                               args.search_max, args.smooth_radius, tmpdir), flush=True)
    print(f'\nwrote previews to {args.outdir}')


if __name__ == '__main__':
    main()
