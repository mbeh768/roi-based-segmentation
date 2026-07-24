"""
Does the FFT axis-notch preprocessing actually change what Otsu decides?

Runs the same Otsu threshold twice per slide -- once on the raw red channel,
once on the axis-notch-filtered red channel (see fft_axis_notch_preview.py
for what that filter does and why) -- and reports both thresholds, both
tissue fractions, and how much the two resulting masks actually agree
(intersection over union). The point is to measure whether the preprocessing
moves the decision at all, not to assume it does.

Both passes exclude completely black pixels (R=G=B=0: scanner/stitching
artifact, not background -- see epicardium_otsu_preview.py). That "real
pixel" mask is computed from the ORIGINAL image in both cases, because the
filter's ringing spreads a little energy into those artifact pixels and they
are no longer exactly zero afterwards; re-deriving the mask post-filter
would silently change which pixels are being compared and make the two runs
non-comparable.

Reuses the full-res TIFF loading from epicardium_otsu_preview.py and the
notch filter from fft_axis_notch_preview.py.

Writes one 4-panel PNG per case (raw | Otsu on raw | filtered | Otsu on
filtered, tissue tinted magenta) to fft_then_otsu_previews/ in the project
root.

Usage:
    python quant/fft_then_otsu_preview.py
    python quant/fft_then_otsu_preview.py --exclude-radius 40
"""
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

from epicardium_otsu_preview import CASES, RGB_ROOT, DEFAULT_MAX_DIM, load_rgb_downsampled
from fft_axis_notch_preview import compute_fft, build_cross_notch_mask

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / 'fft_then_otsu_previews'
DEFAULT_EXCLUDE_RADIUS = 8
DEFAULT_BAND_HALFWIDTH = 2

TISSUE_COLOR = (255, 0, 255)  # BGR magenta
ALPHA = 0.45


def otsu_on(channel, real):
    """Otsu threshold over `real` pixels only. Returns (mask, threshold)."""
    if not real.any():
        return np.zeros(channel.shape, dtype=bool), 0
    thresh, _ = cv2.threshold(channel[real].reshape(-1, 1), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return real & (channel > thresh), int(thresh)


def axis_notch(channel, exclude_radius, band_halfwidth):
    spectrum = compute_fft(channel)
    mask = build_cross_notch_mask(spectrum.shape, exclude_radius, band_halfwidth)
    filtered = np.fft.ifft2(np.fft.ifftshift(spectrum * mask)).real
    return np.clip(filtered, 0, 255).astype(np.uint8)


def tint(gray, mask, color=TISSUE_COLOR, alpha=ALPHA):
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
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


def process_case(stem, rel_path, outdir, max_dim, exclude_radius, band_halfwidth, tmpdir):
    src = RGB_ROOT / rel_path
    if not src.exists():
        return f'{stem}: NOT FOUND -- skipped', None

    case_tmp = tmpdir / stem
    case_tmp.mkdir(parents=True, exist_ok=True)
    bgr, native_hw = load_rgb_downsampled(src, max_dim, case_tmp)
    for f in case_tmp.iterdir():
        f.unlink()
    case_tmp.rmdir()

    red = bgr[..., 2]
    real = np.any(bgr > 0, axis=-1)  # from the ORIGINAL -- see module docstring

    mask_raw, thresh_raw = otsu_on(red, real)
    filtered = axis_notch(red, exclude_radius, band_halfwidth)
    mask_filt, thresh_filt = otsu_on(filtered, real)

    union = (mask_raw | mask_filt).sum()
    iou = (mask_raw & mask_filt).sum() / union if union else 1.0
    frac_raw = 100 * mask_raw.sum() / max(1, real.sum())
    frac_filt = 100 * mask_filt.sum() / max(1, real.sum())

    panels = [
        label_panel(red, f'{stem}  |  raw red channel'),
        label_panel(tint(red, mask_raw), f'otsu on raw: thresh={thresh_raw}, tissue={frac_raw:.1f}%'),
        label_panel(filtered, f'axis-notch filtered (r={exclude_radius})'),
        label_panel(tint(filtered, mask_filt), f'otsu on filtered: thresh={thresh_filt}, tissue={frac_filt:.1f}%'),
    ]
    cv2.imwrite(str(outdir / f'{stem}.png'), np.hstack(panels))

    line = (f'{stem:22s} thresh {thresh_raw:>3} -> {thresh_filt:>3}   '
            f'tissue {frac_raw:>5.1f}% -> {frac_filt:>5.1f}%   IoU {iou:.3f}')
    return line, (thresh_raw, thresh_filt, frac_raw, frac_filt, iou)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument('--max-dim', type=int, default=DEFAULT_MAX_DIM)
    ap.add_argument('--exclude-radius', type=int, default=DEFAULT_EXCLUDE_RADIUS)
    ap.add_argument('--band-halfwidth', type=int, default=DEFAULT_BAND_HALFWIDTH)
    args = ap.parse_args()

    if args.outdir.exists():
        for f in args.outdir.glob('*.png'):
            f.unlink()
    args.outdir.mkdir(parents=True, exist_ok=True)

    stats = []
    with tempfile.TemporaryDirectory(prefix='fft_otsu_vips_') as tmp:
        tmpdir = Path(tmp)
        for stem, rel_path in CASES.items():
            line, s = process_case(stem, rel_path, args.outdir, args.max_dim,
                                   args.exclude_radius, args.band_halfwidth, tmpdir)
            print(line, flush=True)
            if s:
                stats.append(s)

    if stats:
        arr = np.array(stats)
        print(f'\nmedian |thresh change|: {np.median(np.abs(arr[:,1]-arr[:,0])):.1f}')
        print(f'median |tissue%% change|: {np.median(np.abs(arr[:,3]-arr[:,2])):.2f}')
        print(f'median IoU raw-vs-filtered: {np.median(arr[:,4]):.3f}')
    print(f'\nwrote previews to {args.outdir}')


if __name__ == '__main__':
    main()
