"""
Step 1 of a Laplacian-of-Gaussian (LoG) based approach: just generate and look
at the LoG response of the red channel (cardiomyocyte stain) at sigma=30,
before deciding what to threshold or filter on top of it.

WHAT THIS IS: the Laplacian of Gaussian is the image convolved with the
second derivative of a Gaussian kernel. Concretely, at each pixel it answers
"compared to a Gaussian-weighted average of the surroundings out to about
`sigma` pixels, is this point sitting in a locally brighter blob, a locally
darker hole, or on a fairly flat/linear region?" -- it's near zero on flat
regions and smooth gradients, strongly negative in the middle of bright
blobs of about that scale, and strongly positive in the middle of dark
blobs/holes of about that scale. sigma=30 sets the scale being probed: it
responds to structure roughly 30 pixels across (at this script's preview
resolution, downsampled from the full-res slide -- see DEFAULT_MAX_DIM),
and is blind to anything much smaller (fine noise) or much larger (the
slide's overall brightness gradient).

This uses scipy.ndimage.gaussian_laplace, which convolves directly with the
analytic second-derivative-of-Gaussian kernel (separable, sum of the two
axis-aligned second derivatives) -- not an approximation via a Gaussian blur
followed by a separate Laplacian operator.

Reuses the full-res TIFF loading from epicardium_otsu_preview.py (the CASES
path table and the vips-based downsampler that handles this dataset's three
on-disk layouts) rather than re-deriving it.

Writes one 2-panel PNG per case (original red channel | LoG response,
min/max labeled) to laplacian_gaussian_previews/ in the project root. Does
not touch any of the other preview/histogram folders from earlier
diagnostic scripts.

Usage:
    python quant/laplacian_gaussian_preview.py
    python quant/laplacian_gaussian_preview.py --sigma 30
"""
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_laplace

from epicardium_otsu_preview import CASES, RGB_ROOT, DEFAULT_MAX_DIM, load_rgb_downsampled

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SIGMA = 30


def default_outdir(sigma):
    """One folder per sigma (e.g. laplacian_gaussian_previews_sigma15) --
    different scales are different, non-comparable diagnostics, and a shared
    folder would silently overwrite one scale's previews with another's."""
    sigma_label = f'{sigma:g}'.replace('.', '_')
    return REPO_ROOT / f'laplacian_gaussian_previews_sigma{sigma_label}'


def normalize_for_display(response):
    """LoG output is signed (negative in bright blobs, positive in dark
    holes/background) and has no fixed range, so linearly map this
    particular image's [min, max] to [0, 255] uint8 for viewing. Purely a
    display transform -- the underlying `response` array (used for anything
    downstream) is left as signed floats."""
    lo, hi = float(response.min()), float(response.max())
    if hi <= lo:
        return np.zeros(response.shape, dtype=np.uint8), lo, hi
    scaled = (response - lo) / (hi - lo) * 255.0
    return scaled.astype(np.uint8), lo, hi


def label_panel(image, text):
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(bgr, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def process_case(stem, rel_path, outdir, max_dim, sigma, tmpdir):
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
    log_response = gaussian_laplace(red.astype(np.float64), sigma=sigma)
    display, lo, hi = normalize_for_display(log_response)

    p1 = label_panel(red, f'{stem}  |  native {native_hw[1]}x{native_hw[0]}  |  original red channel')
    p2 = label_panel(display, f'LoG response, sigma={sigma}  (range [{lo:.1f}, {hi:.1f}] mapped to 0-255)')

    out_path = outdir / f'{stem}.png'
    cv2.imwrite(str(out_path), np.hstack([p1, p2]))
    return f'{stem}: native {native_hw[1]}x{native_hw[0]}, LoG range [{lo:.1f}, {hi:.1f}], wrote {out_path.name}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', type=Path, default=None,
                    help='default: laplacian_gaussian_previews_sigma<N> (one folder per sigma)')
    ap.add_argument('--max-dim', type=int, default=DEFAULT_MAX_DIM)
    ap.add_argument('--sigma', type=float, default=DEFAULT_SIGMA)
    args = ap.parse_args()
    if args.outdir is None:
        args.outdir = default_outdir(args.sigma)

    if args.outdir.exists():
        for f in args.outdir.glob('*.png'):
            f.unlink()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='log_preview_vips_') as tmp:
        tmpdir = Path(tmp)
        for stem, rel_path in CASES.items():
            print(process_case(stem, rel_path, args.outdir, args.max_dim, args.sigma, tmpdir),
                  flush=True)
    print(f'\nwrote previews to {args.outdir}')


if __name__ == '__main__':
    main()
