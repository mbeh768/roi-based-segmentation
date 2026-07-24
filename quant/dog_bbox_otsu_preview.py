"""
Difference of a 2nd-order and a 1st-order Gaussian derivative, used to locate
the tissue and restrict Otsu to a bounding box around it.

THE RESPONSE (user's recipe, both at sigma=6):
    response = LoG(sigma=6)  -  gradientMagnitude(sigma=6)
where LoG is the Laplacian of Gaussian (scipy.ndimage.gaussian_laplace, the
2nd derivative, signed: negative in bright blobs, positive in dark holes) and
gradientMagnitude is the first-order Gaussian derivative magnitude
(scipy.ndimage.gaussian_gradient_magnitude, |grad|, always >= 0). Subtracting
the edge term from the blob term suppresses pure step edges (which score high
in |grad| but near-zero in LoG) while keeping blob-like interiors, so the
response concentrates on tissue body rather than on every boundary.

BOUNDING BOX + OTSU:
  1. threshold |response| (Otsu over real, non-artifact pixels) to get a
     coarse tissue-presence mask;
  2. keep connected components >= BBOX_ISLAND_FRAC of the largest, so stray
     specks don't inflate the box;
  3. take the bounding box of what survives;
  4. run Otsu on the RAW red channel using only real pixels inside that box,
     and compare the resulting threshold to plain full-frame Otsu.

The point of the comparison panel is to make it obvious whether restricting
to the box actually changes the threshold or not -- earlier analysis suggests
it may be close to a no-op, since full-frame Otsu already ignores the
zero-valued margins the box removes.

Reuses the full-res TIFF loading from epicardium_otsu_preview.py (the CASES
path table and the vips-based downsampler that handles this dataset's three
on-disk layouts).

Writes one 6-panel PNG per case (raw | LoG | gradient mag | difference |
bbox on tissue-presence mask | raw with in-box Otsu tissue tinted, box drawn)
to dog_bbox_otsu_previews/ in the project root.

Usage:
    python quant/dog_bbox_otsu_preview.py
    python quant/dog_bbox_otsu_preview.py --sigma 6
"""
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_laplace, gaussian_gradient_magnitude

from epicardium_otsu_preview import CASES, RGB_ROOT, DEFAULT_MAX_DIM, load_rgb_downsampled

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / 'dog_bbox_otsu_previews'
DEFAULT_SIGMA = 6
BBOX_ISLAND_FRAC = 0.05       # drop presence-mask components <= this fraction of the largest
TISSUE_COLOR = (0, 255, 0)    # BGR green
BOX_COLOR = (0, 165, 255)     # BGR orange
ALPHA = 0.40


def dog_response(red, sigma):
    """LoG(sigma) - gradientMagnitude(sigma). Both computed on float; the
    result is signed and unbounded, returned as float."""
    log = gaussian_laplace(red.astype(np.float64), sigma=sigma)
    grad = gaussian_gradient_magnitude(red.astype(np.float64), sigma=sigma)
    return log - grad


def signed_to_display(arr):
    """Map a signed, arbitrary-range array to 0-255 for viewing (its own
    min->0, max->255). Display only."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)


def presence_mask(response, real, island_frac=BBOX_ISLAND_FRAC):
    """Coarse 'tissue is here' mask: Otsu on |response| over real pixels, then
    keep only sizable connected components."""
    mag = np.abs(response)
    mag8 = signed_to_display(mag)
    if not real.any():
        return np.zeros(response.shape, bool)
    thresh, _ = cv2.threshold(mag8[real].reshape(-1, 1), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coarse = real & (mag8 > thresh)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        coarse.astype(np.uint8), connectivity=8)
    if n <= 2:
        return coarse
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, np.uint8)
    keep[1:] = (areas > island_frac * areas.max()).astype(np.uint8)
    return keep[labels].astype(bool)


def bounding_box(mask):
    """(x0, y0, x1, y1) inclusive-exclusive box of the True region, or None."""
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def otsu_on(channel, real):
    if not real.any():
        return np.zeros(channel.shape, bool), 0
    thresh, _ = cv2.threshold(channel[real].reshape(-1, 1), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return real & (channel > thresh), int(thresh)


def tint(gray_or_bgr, mask, color=TISSUE_COLOR, alpha=ALPHA):
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


def process_case(stem, rel_path, outdir, max_dim, sigma, tmpdir):
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

    log = gaussian_laplace(red.astype(np.float64), sigma=sigma)
    grad = gaussian_gradient_magnitude(red.astype(np.float64), sigma=sigma)
    response = log - grad

    presence = presence_mask(response, real)
    box = bounding_box(presence)

    # full-frame Otsu for comparison, then in-box Otsu
    _, thresh_full = otsu_on(red, real)
    if box is not None:
        x0, y0, x1, y1 = box
        in_box = np.zeros(real.shape, bool)
        in_box[y0:y1, x0:x1] = True
        real_box = real & in_box
        mask_box, thresh_box = otsu_on(red, real_box)
    else:
        mask_box, thresh_box, box = np.zeros(real.shape, bool), 0, (0, 0, red.shape[1], red.shape[0])

    frac = 100 * mask_box.sum() / max(1, real.sum())

    # presence panel with the bbox drawn
    presence_panel = cv2.cvtColor((presence * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(presence_panel, (box[0], box[1]), (box[2] - 1, box[3] - 1), BOX_COLOR, 2)
    final = tint(red, mask_box)
    cv2.rectangle(final, (box[0], box[1]), (box[2] - 1, box[3] - 1), BOX_COLOR, 2)

    panels = [
        label_panel(red, f'{stem}  |  1. raw red  (native {native_hw[1]}x{native_hw[0]})'),
        label_panel(signed_to_display(log), f'2. LoG sigma={sigma}'),
        label_panel(signed_to_display(grad), f'3. gradient mag sigma={sigma}'),
        label_panel(signed_to_display(response), '4. difference (LoG - grad)'),
        label_panel(presence_panel, '5. tissue-presence + bbox'),
        label_panel(final, f'6. in-box otsu={thresh_box} (full={thresh_full})  tissue {frac:.1f}%'),
    ]
    cv2.imwrite(str(outdir / f'{stem}.png'), np.hstack(panels))
    return f'{stem}: otsu full={thresh_full:>3} in-box={thresh_box:>3}   tissue {frac:5.1f}% of real'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument('--max-dim', type=int, default=DEFAULT_MAX_DIM)
    ap.add_argument('--sigma', type=float, default=DEFAULT_SIGMA)
    args = ap.parse_args()

    if args.outdir.exists():
        for f in args.outdir.glob('*.png'):
            f.unlink()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='dog_bbox_vips_') as tmp:
        tmpdir = Path(tmp)
        for stem, rel_path in CASES.items():
            print(process_case(stem, rel_path, args.outdir, args.max_dim, args.sigma, tmpdir),
                  flush=True)
    print(f'\nwrote previews to {args.outdir}')


if __name__ == '__main__':
    main()
