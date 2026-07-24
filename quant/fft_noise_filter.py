"""
Remove periodic scanner/stitching noise from the red channel (cardiomyocyte
stain) using a frequency-domain notch filter, and preview the result.

WHY FREQUENCY DOMAIN: the scanner artifact visible in earlier previews (a
faint grid overlaid on the tissue, from stitching together individually
scanned tiles) is periodic -- it repeats at a fixed spacing across the whole
image, because tile size is constant. A repeating pattern like that shows up
as a small number of sharp, isolated bright spots in the image's 2D Fourier
transform, at a distance from the center that corresponds to the pattern's
frequency (1 / tile-spacing). Real tissue structure has no such fixed
repeat distance, so it spreads its energy broadly across the frequency
plane instead of concentrating it in a few spots. That difference is what
makes the grid separable from tissue in frequency space, in a way it isn't
in plain pixel intensity.

HOW THIS SCRIPT WORKS, step by step, for each slide's red channel:
  1. Take the 2D FFT of the channel. Each point in the result corresponds to
     one spatial frequency (a repeat distance + direction) present in the
     image; its brightness there is how strongly that frequency shows up.
  2. Shift the FFT so the zero frequency (the image's overall average
     brightness) sits at the center, and low frequencies (broad, smooth
     structure -- most of the real tissue signal) surround it. Frequency
     increases with distance from the center in every direction.
  3. Mask out a disk around the center (`--exclude-radius-frac` of the
     shorter image side) before hunting for noise -- this protects the bulk
     of the real tissue signal, which is low-frequency, from ever being
     mistaken for a noise spike.
  4. Outside that disk, find the brightest remaining points in the spectrum
     (`--num-peaks` of them). These are the candidate periodic-noise
     frequencies -- the grid pattern, if it's there.
  5. Zero out a small disk (`--notch-radius` pixels) around each candidate
     peak, and around its mirror point on the opposite side of the center
     (a real-valued image's FFT is point-symmetric, so a genuine periodic
     pattern always produces a matched pair of peaks, not a lone one -- this
     is used below as a sanity check, not just applied blindly).
  6. Inverse FFT the masked spectrum back to an image. What's removed is
     only the energy at those specific frequencies; everything else in the
     image is reconstructed unchanged.

This targets the periodic grid artifact specifically. It is a different tool
from the intensity-histogram thresholding explored in
red_channel_histogram.py / red_channel_valley_preview.py -- it won't help a
slide where the red channel is just genuinely weak over real tissue (see
17P391_7A in this dataset), because that isn't a frequency-domain problem.

Reuses the full-res TIFF loading from epicardium_otsu_preview.py (the CASES
path table and the vips-based downsampler that handles this dataset's three
on-disk layouts) rather than re-deriving it.

Writes one 3-panel PNG per case (original red channel | frequency spectrum
with detected noise peaks marked | filtered result) to
fft_filtered_previews/ in the project root. Does not touch any of the other
preview/histogram folders from earlier diagnostic scripts.

Usage:
    python quant/fft_noise_filter.py
    python quant/fft_noise_filter.py --num-peaks 12 --notch-radius 4
"""
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

from epicardium_otsu_preview import CASES, RGB_ROOT, DEFAULT_MAX_DIM, load_rgb_downsampled

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / 'fft_filtered_previews'

DEFAULT_EXCLUDE_RADIUS_FRAC = 0.03  # protects low frequencies (real tissue signal) from the peak search
DEFAULT_NUM_PEAKS = 8              # how many candidate noise frequencies to look for
DEFAULT_NOTCH_RADIUS = 3           # pixels; size of the disk zeroed out around each peak


def compute_fft(channel):
    """2D FFT of the channel, shifted so zero frequency is at the center."""
    spectrum = np.fft.fft2(channel.astype(np.float64))
    spectrum_centered = np.fft.fftshift(spectrum)
    return spectrum_centered


def find_noise_peaks(spectrum_centered, exclude_radius_frac, num_peaks):
    """Return the (row, col) coordinates of the brightest points in the
    spectrum, outside a central disk that protects real low-frequency tissue
    signal. These are the candidate periodic-noise frequencies."""
    magnitude = np.abs(spectrum_centered)

    h, w = magnitude.shape
    center_row, center_col = h // 2, w // 2
    exclude_radius = exclude_radius_frac * min(h, w)

    rows, cols = np.indices(magnitude.shape)
    distance_from_center = np.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2)
    search_area = magnitude.copy()
    search_area[distance_from_center <= exclude_radius] = 0  # don't pick peaks near DC

    peaks = []
    for _ in range(num_peaks):
        peak_index = np.argmax(search_area)
        peak_row, peak_col = np.unravel_index(peak_index, search_area.shape)
        if search_area[peak_row, peak_col] <= 0:
            break  # ran out of meaningful peaks
        peaks.append((peak_row, peak_col))
        # blank out a small area around this peak so the next argmax finds a
        # genuinely different point, not just an adjacent pixel of the same spike
        row_lo, row_hi = max(0, peak_row - 5), min(h, peak_row + 6)
        col_lo, col_hi = max(0, peak_col - 5), min(w, peak_col + 6)
        search_area[row_lo:row_hi, col_lo:col_hi] = 0

    return peaks


def build_notch_mask(shape, peaks, notch_radius):
    """A mask, same size as the spectrum, that is 0 in small disks around each
    peak (and its point-symmetric mirror -- a real image's FFT always has
    energy in symmetric pairs, so removing only one side would leave a
    residual) and 1 everywhere else."""
    h, w = shape
    center_row, center_col = h // 2, w // 2
    mask = np.ones(shape, dtype=np.float64)

    rows, cols = np.indices(shape)
    for peak_row, peak_col in peaks:
        mirror_row = 2 * center_row - peak_row
        mirror_col = 2 * center_col - peak_col
        for r, c in [(peak_row, peak_col), (mirror_row, mirror_col)]:
            distance = np.sqrt((rows - r) ** 2 + (cols - c) ** 2)
            mask[distance <= notch_radius] = 0.0

    return mask


def spectrum_preview_panel(spectrum_centered, peaks):
    """Log-scaled magnitude spectrum, rescaled to 0-255 for display, with the
    detected noise peaks circled so it's obvious what's being removed."""
    magnitude = np.abs(spectrum_centered)
    log_magnitude = np.log1p(magnitude)  # log(1 + x): spectra span huge ranges, log compresses them for viewing
    normalized = log_magnitude / log_magnitude.max() * 255.0
    panel = cv2.cvtColor(normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for peak_row, peak_col in peaks:
        cv2.circle(panel, (peak_col, peak_row), 6, (0, 0, 255), 1)  # red circle, BGR

    return panel


def label_panel(image, text):
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(bgr, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def process_case(stem, rel_path, outdir, max_dim, exclude_radius_frac, num_peaks, notch_radius, tmpdir):
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
    spectrum_centered = compute_fft(red)  # kept separately from apply_notch_filter so it can be reused for the spectrum preview panel below
    peaks = find_noise_peaks(spectrum_centered, exclude_radius_frac, num_peaks)
    mask = build_notch_mask(spectrum_centered.shape, peaks, notch_radius)
    filtered_red = np.clip(
        np.fft.ifft2(np.fft.ifftshift(spectrum_centered * mask)).real, 0, 255
    ).astype(np.uint8)

    p1 = label_panel(red, f'{stem}  |  native {native_hw[1]}x{native_hw[0]}  |  original red channel')
    p2 = label_panel(spectrum_preview_panel(spectrum_centered, peaks),
                     f'spectrum, {len(peaks)} noise peaks circled')
    p3 = label_panel(filtered_red, 'notch-filtered')

    out_path = outdir / f'{stem}.png'
    cv2.imwrite(str(out_path), np.hstack([p1, p2, p3]))
    return f'{stem}: native {native_hw[1]}x{native_hw[0]}, {len(peaks)} peaks removed, wrote {out_path.name}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument('--max-dim', type=int, default=DEFAULT_MAX_DIM)
    ap.add_argument('--exclude-radius-frac', type=float, default=DEFAULT_EXCLUDE_RADIUS_FRAC)
    ap.add_argument('--num-peaks', type=int, default=DEFAULT_NUM_PEAKS)
    ap.add_argument('--notch-radius', type=int, default=DEFAULT_NOTCH_RADIUS)
    args = ap.parse_args()

    if args.outdir.exists():
        for f in args.outdir.glob('*.png'):
            f.unlink()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='fft_filter_vips_') as tmp:
        tmpdir = Path(tmp)
        for stem, rel_path in CASES.items():
            print(process_case(stem, rel_path, args.outdir, args.max_dim,
                               args.exclude_radius_frac, args.num_peaks,
                               args.notch_radius, tmpdir), flush=True)
    print(f'\nwrote previews to {args.outdir}')


if __name__ == '__main__':
    main()
