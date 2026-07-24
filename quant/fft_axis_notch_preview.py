"""
Remove the scanner/stitching grid artifact with a direct frequency-domain
cross notch -- no peak detection, no heuristics.

WHY A CROSS, NOT PEAK-HUNTING: the grid comes from stitching together
individually scanned rectangular tiles, so its seams are straight lines
running horizontally and vertically across the whole image, repeating at a
fixed spacing (tile size). A pattern that repeats periodically along the X
direction (vertical seams, repeating as you move horizontally) puts all of
its energy on the *horizontal* frequency axis of the image's 2D FFT (the
center row, after shifting zero-frequency to the middle); a pattern
repeating along Y (horizontal seams) puts its energy on the *vertical*
frequency axis (the center column). A grid has both, so its energy sits
squarely on a cross through the center of the spectrum -- this is true
regardless of the exact tile size or how many harmonics the (non-sinusoidal,
roughly square-wave-shaped) seam pattern produces, since every harmonic of
an axis-aligned repeat still lands somewhere on that same cross.

fft_noise_filter.py's earlier approach (hunting for the brightest points
anywhere in the 2D spectrum) didn't use this structure -- it could just as
easily lock onto strong directional tissue texture as onto the grid, and had
no reason to find every harmonic. Restricting the notch to the cross is both
simpler and more targeted: real tissue structure is essentially never
perfectly axis-aligned and periodic the way a stitching seam is, so very
little real signal lives exactly on that cross, and every point on it is
fair game.

HOW THIS SCRIPT WORKS, for each slide's red channel:
  1. Take the 2D FFT, shifted so zero frequency (the channel's average
     brightness) sits at the center and frequency increases with distance
     from the center in every direction.
  2. Protect a small disk around the center (`--exclude-radius`, in FFT
     pixels) from any modification -- this is the real low-frequency tissue
     signal (the slide's overall shape and broad brightness gradients),
     which is legitimate and must not be touched even though it sits near
     the axes too.
  3. Outside that disk, zero out a thin band (`--band-halfwidth` pixels
     either side) along the horizontal and vertical frequency axes -- the
     cross. Everything off-axis is left completely alone.
  4. Inverse FFT back to an image.

Reuses the full-res TIFF loading from epicardium_otsu_preview.py (the CASES
path table and the vips-based downsampler that handles this dataset's three
on-disk layouts) rather than re-deriving it.

Writes one 3-panel PNG per case (original red channel | spectrum with the
notched cross marked | filtered result) to fft_axis_notch_previews/ in the
project root. Does not touch any of the other preview/histogram folders from
earlier diagnostic scripts, including fft_noise_filter.py's
fft_filtered_previews/.

Usage:
    python quant/fft_axis_notch_preview.py
    python quant/fft_axis_notch_preview.py --exclude-radius 8 --band-halfwidth 2
"""
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

from epicardium_otsu_preview import CASES, RGB_ROOT, DEFAULT_MAX_DIM, load_rgb_downsampled

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / 'fft_axis_notch_previews'

DEFAULT_EXCLUDE_RADIUS = 8    # FFT pixels; protects real low-frequency tissue signal near DC
DEFAULT_BAND_HALFWIDTH = 2    # FFT pixels; width of the notched band on each side of each axis


def compute_fft(channel):
    """2D FFT of the channel, shifted so zero frequency is at the center."""
    spectrum = np.fft.fft2(channel.astype(np.float64))
    return np.fft.fftshift(spectrum)


def build_cross_notch_mask(shape, exclude_radius, band_halfwidth):
    """1 everywhere except a thin band along the horizontal and vertical
    frequency axes (0 there), with a disk around the center exempted so the
    real low-frequency signal near DC is never touched."""
    h, w = shape
    center_row, center_col = h // 2, w // 2
    rows, cols = np.indices(shape)

    on_horizontal_axis = np.abs(rows - center_row) <= band_halfwidth  # kills vertical-seam repeats (period along Y)
    on_vertical_axis = np.abs(cols - center_col) <= band_halfwidth    # kills horizontal-seam repeats (period along X)
    on_cross = on_horizontal_axis | on_vertical_axis

    distance_from_center = np.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2)
    protected = distance_from_center <= exclude_radius

    mask = np.ones(shape, dtype=np.float64)
    mask[on_cross & ~protected] = 0.0
    return mask


def spectrum_preview_panel(spectrum_centered, mask):
    """Log-scaled magnitude spectrum, rescaled to 0-255, with the notched
    (removed) region tinted red so it's obvious what's being cut."""
    magnitude = np.abs(spectrum_centered)
    log_magnitude = np.log1p(magnitude)
    normalized = log_magnitude / log_magnitude.max() * 255.0
    gray = normalized.astype(np.uint8)
    panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    removed = mask == 0.0
    panel[removed] = (panel[removed] * 0.3 + np.array([0, 0, 255]) * 0.7).astype(np.uint8)  # BGR red tint
    return panel


def label_panel(image, text):
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(bgr, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def process_case(stem, rel_path, outdir, max_dim, exclude_radius, band_halfwidth, tmpdir):
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
    spectrum_centered = compute_fft(red)
    mask = build_cross_notch_mask(spectrum_centered.shape, exclude_radius, band_halfwidth)
    filtered_red = np.clip(
        np.fft.ifft2(np.fft.ifftshift(spectrum_centered * mask)).real, 0, 255
    ).astype(np.uint8)

    p1 = label_panel(red, f'{stem}  |  native {native_hw[1]}x{native_hw[0]}  |  original red channel')
    p2 = label_panel(spectrum_preview_panel(spectrum_centered, mask),
                     f'spectrum, notched cross in red (exclude r={exclude_radius}, band={band_halfwidth})')
    p3 = label_panel(filtered_red, 'axis-notch filtered')

    out_path = outdir / f'{stem}.png'
    cv2.imwrite(str(out_path), np.hstack([p1, p2, p3]))
    return f'{stem}: native {native_hw[1]}x{native_hw[0]}, wrote {out_path.name}'


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

    with tempfile.TemporaryDirectory(prefix='fft_axis_notch_vips_') as tmp:
        tmpdir = Path(tmp)
        for stem, rel_path in CASES.items():
            print(process_case(stem, rel_path, args.outdir, args.max_dim,
                               args.exclude_radius, args.band_halfwidth, tmpdir), flush=True)
    print(f'\nwrote previews to {args.outdir}')


if __name__ == '__main__':
    main()
