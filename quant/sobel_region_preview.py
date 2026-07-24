"""
Edge-based tissue regions: find the tissue BOUNDARY with a first-order
Gaussian derivative, close it into a ring, fill it, and overlay the result on
a gamma-brightened original.

WHY EDGES RATHER THAN INTENSITY: every intensity-threshold attempt so far
(plain Otsu, valley-emphasis, percentile clipping, noise-floor estimation)
has been defeated by the same thing -- tissue brightness varies enormously
both within a slide and between slides, so no single intensity value
separates tissue from background everywhere. The tissue/background
*boundary*, though, is a brightness discontinuity regardless of how bright
the tissue is on either side of it. An edge detector keys on the change
rather than the level, which is why it can work where a level-based
threshold cannot.

THE STEPS (one preview panel each, left to right):

  1. raw red channel (cardiomyocyte stain).

  2. first-order Gaussian derivative magnitude at sigma=8, via
     scipy.ndimage.gaussian_gradient_magnitude. This convolves with the
     first derivative of a Gaussian in x and y and takes the magnitude
     sqrt(gx^2 + gy^2) -- the same quantity a Sobel operator estimates, but
     at a controlled scale instead of Sobel's fixed 3x3 kernel. sigma=8
     means it responds to brightness transitions spread over roughly 8
     pixels and ignores finer variation, so tissue texture and the
     stitching-seam steps are largely smoothed past while the outer
     tissue/background boundary still registers. (This is the first-order
     counterpart to laplacian_gaussian_preview.py's second-order operator.)

  3. edge mask: Otsu threshold on that gradient magnitude. Note this is Otsu
     on the GRADIENT, not on intensity -- the failure mode that broke the
     intensity thresholds does not apply, because a strong boundary is
     strong in gradient terms whether the tissue behind it is bright or dim.

  4. closed regions: the edge mask is a ring (or a broken arc) around the
     tissue, not a filled shape. Morphological closing seals gaps where the
     boundary faded, binary_fill_holes floods the interior, and connected
     components smaller than ISLAND_FRAC of the largest are dropped.

  5. gamma-brightened original (gamma 0.4, as in
     cardiomyocyte_gamma_preview.py) with the filled region tinted, so dim
     tissue is actually visible when judging whether the region got it.

Reuses the full-res TIFF loading from epicardium_otsu_preview.py (the CASES
path table and the vips-based downsampler that handles this dataset's three
on-disk layouts) rather than re-deriving it.

Writes one 5-panel PNG per case to sobel_region_previews/ in the project
root.

Usage:
    python quant/sobel_region_preview.py
    python quant/sobel_region_preview.py --sigma 8 --close-radius 5
"""
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_gradient_magnitude

from epicardium_otsu_preview import CASES, RGB_ROOT, DEFAULT_MAX_DIM, load_rgb_downsampled

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SIGMA = 20


def default_outdir(sigma):
    """One folder per sigma (e.g. sobel_region_previews_sigma20) -- different
    scales are different, non-comparable diagnostics, and a shared folder
    would silently overwrite one scale's previews with another's. Same
    convention as laplacian_gaussian_preview.py."""
    return REPO_ROOT / f"sobel_region_previews_sigma{f'{sigma:g}'.replace('.', '_')}"
DEFAULT_CLOSE_RADIUS = 5   # px; seals gaps in the boundary ring before filling
DEFAULT_GAMMA = 0.4
ISLAND_FRAC = 0.05         # drop filled components <= this fraction of the largest
REGION_COLOR = (0, 255, 0)  # BGR green
ALPHA = 0.40


def gradient_magnitude(red, sigma):
    """First-order Gaussian derivative magnitude -- the Sobel quantity at a
    controlled scale. Returned as float; normalized separately for display."""
    return gaussian_gradient_magnitude(red.astype(np.float64), sigma=sigma)


def edge_mask_from_gradient(grad, real):
    """Otsu on the gradient magnitude (over real pixels only), keeping the
    high-gradient class -- i.e. the boundary."""
    g8 = np.clip(grad / grad.max() * 255.0, 0, 255).astype(np.uint8) if grad.max() > 0 \
        else np.zeros(grad.shape, np.uint8)
    if not real.any():
        return np.zeros(grad.shape, bool), g8
    thresh, _ = cv2.threshold(g8[real].reshape(-1, 1), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return real & (g8 > thresh), g8


def close_and_fill(edges, close_radius, island_frac=ISLAND_FRAC):
    """Turn a boundary ring into a filled region: seal gaps, flood the
    interior, drop small stray components."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1,) * 2)
    sealed = cv2.morphologyEx(edges.astype(np.uint8), cv2.MORPH_CLOSE, k)
    filled = binary_fill_holes(sealed.astype(bool))
    if filled is None or not filled.any():
        return np.zeros(edges.shape, bool)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        filled.astype(np.uint8), connectivity=8)
    if n <= 2:
        return filled
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, np.uint8)
    keep[1:] = (areas > island_frac * areas.max()).astype(np.uint8)
    return keep[labels].astype(bool)


def apply_gamma(gray, gamma):
    lut = (np.linspace(0, 1, 256) ** gamma * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.LUT(gray, lut)


def tint(gray_or_bgr, mask, color=REGION_COLOR, alpha=ALPHA):
    bgr = cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2BGR) if gray_or_bgr.ndim == 2 \
        else gray_or_bgr.copy()
    if mask.any():
        patch = bgr[mask].astype(np.float32)
        bgr[mask] = np.clip(patch * (1 - alpha) + np.array(color, np.float32) * alpha,
                            0, 255).astype(np.uint8)
    return bgr


def label_panel(image, text):
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(bgr, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def process_case(stem, rel_path, outdir, max_dim, sigma, close_radius, gamma, tmpdir):
    src = RGB_ROOT / rel_path
    if not src.exists():
        return f'{stem}: NOT FOUND -- skipped'

    case_tmp = tmpdir / stem
    case_tmp.mkdir(parents=True, exist_ok=True)
    bgr, native_hw = load_rgb_downsampled(src, max_dim, case_tmp)
    for f in case_tmp.iterdir():
        f.unlink()
    case_tmp.rmdir()

    red = bgr[..., 2]
    real = np.any(bgr > 0, axis=-1)

    grad = gradient_magnitude(red, sigma)
    edges, grad_display = edge_mask_from_gradient(grad, real)
    regions = close_and_fill(edges, close_radius)
    brightened = apply_gamma(red, gamma)

    frac = 100 * regions.sum() / max(1, real.sum())
    panels = [
        label_panel(red, f'{stem}  |  1. raw red  (native {native_hw[1]}x{native_hw[0]})'),
        label_panel(grad_display, f'2. gaussian gradient mag, sigma={sigma}'),
        label_panel((edges * 255).astype(np.uint8), '3. edge mask (otsu on gradient)'),
        label_panel((regions * 255).astype(np.uint8), f'4. closed+filled regions (r={close_radius})'),
        label_panel(tint(brightened, regions), f'5. gamma {gamma} + region  ({frac:.1f}% of real)'),
    ]
    cv2.imwrite(str(outdir / f'{stem}.png'), np.hstack(panels))
    return f'{stem}: region {frac:5.1f}% of real  (edges {100*edges.sum()/max(1,real.sum()):4.1f}%)'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', type=Path, default=None,
                    help='default: sobel_region_previews_sigma<N> (one folder per sigma)')
    ap.add_argument('--max-dim', type=int, default=DEFAULT_MAX_DIM)
    ap.add_argument('--sigma', type=float, default=DEFAULT_SIGMA)
    ap.add_argument('--close-radius', type=int, default=DEFAULT_CLOSE_RADIUS)
    ap.add_argument('--gamma', type=float, default=DEFAULT_GAMMA)
    args = ap.parse_args()
    if args.outdir is None:
        args.outdir = default_outdir(args.sigma)

    if args.outdir.exists():
        for f in args.outdir.glob('*.png'):
            f.unlink()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='sobel_region_vips_') as tmp:
        tmpdir = Path(tmp)
        for stem, rel_path in CASES.items():
            print(process_case(stem, rel_path, args.outdir, args.max_dim, args.sigma,
                               args.close_radius, args.gamma, tmpdir), flush=True)
    print(f'\nwrote previews to {args.outdir}')


if __name__ == '__main__':
    main()
