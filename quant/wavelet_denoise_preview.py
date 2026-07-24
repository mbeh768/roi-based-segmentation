"""
Wavelet-domain denoising of the red channel (cardiomyocyte stain), assuming
additive Gaussian noise -- the classical VisuShrink method (Donoho &
Johnstone, 1994).

HOW THIS WORKS: a wavelet transform decomposes the image into an
approximation (coarse, blurry version) plus a set of "detail" coefficients
at several scales, each describing the fine structure the approximation is
missing at that scale. Real image structure (tissue edges, texture) tends to
produce a few large detail coefficients concentrated where the structure is;
additive Gaussian noise, being spatially unstructured, spreads itself out as
many small detail coefficients everywhere. VisuShrink exploits that
difference: estimate the noise's standard deviation from the finest-scale
detail coefficients (robustly, via median absolute deviation -- see
skimage's implementation), then shrink every detail coefficient toward zero
by a threshold set from that noise level (`sigma * sqrt(2 * log(n_pixels))`,
the "universal threshold"). Coefficients dominated by noise get shrunk away
almost entirely; coefficients dominated by real structure survive mostly
intact. Reconstructing from the shrunk coefficients gives back an image with
much of the noise removed and the real structure largely preserved.

This is a different kind of tool from the intensity-histogram thresholding
explored earlier (red_channel_histogram.py etc.) or the frequency-domain
notch filter (fft_noise_filter.py): those try to draw a line that separates
tissue from background/noise; this tries to remove noise from the image
without drawing that line at all, which may make a subsequent threshold step
(on the cleaned channel) easier and more stable.

Uses skimage.restoration.denoise_wavelet(method='VisuShrink') under the
hood (which in turn requires PyWavelets for the actual wavelet transform).

Reuses the full-res TIFF loading from epicardium_otsu_preview.py (the CASES
path table and the vips-based downsampler that handles this dataset's three
on-disk layouts) rather than re-deriving it.

Writes one 3-panel PNG per case (original red channel | denoised | removed
noise, i.e. original minus denoised) to wavelet_denoise_previews/ in the
project root. Does not touch any of the other preview/histogram folders from
earlier diagnostic scripts.

Usage:
    python quant/wavelet_denoise_preview.py
    python quant/wavelet_denoise_preview.py --wavelet db2 --levels 3
"""
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
from skimage.restoration import denoise_wavelet

from epicardium_otsu_preview import CASES, RGB_ROOT, DEFAULT_MAX_DIM, load_rgb_downsampled

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / 'wavelet_denoise_previews'
DEFAULT_WAVELET = 'haar'
DEFAULT_LEVELS = None  # None -> skimage picks (3 less than the max possible)


def denoise_red_channel(red_u8, wavelet, levels):
    """Gaussian-noise wavelet denoising (VisuShrink) on the red channel.
    Works in [0, 1] float, as skimage expects, and converts back to uint8
    for display/saving."""
    red_float = red_u8.astype(np.float64) / 255.0
    denoised_float = denoise_wavelet(
        red_float,
        method='VisuShrink',
        mode='soft',           # soft-thresholding: shrink toward zero rather than hard cutoff, the standard VisuShrink choice
        wavelet=wavelet,
        wavelet_levels=levels,
        rescale_sigma=True,
    )
    denoised_u8 = np.clip(denoised_float * 255.0, 0, 255).astype(np.uint8)
    return denoised_u8


def residual_panel(original_u8, denoised_u8):
    """What the denoiser removed, i.e. the estimated noise, visualized on a
    0-255 scale centered at 128 (so removed-brightness and removed-darkness
    are both visible, not just clipped to one side)."""
    diff = original_u8.astype(np.int16) - denoised_u8.astype(np.int16)
    return np.clip(diff + 128, 0, 255).astype(np.uint8)


def label_panel(image, text):
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(bgr, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def process_case(stem, rel_path, outdir, max_dim, wavelet, levels, tmpdir):
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
    denoised = denoise_red_channel(red, wavelet, levels)
    residual = residual_panel(red, denoised)

    p1 = label_panel(red, f'{stem}  |  native {native_hw[1]}x{native_hw[0]}  |  original red channel')
    p2 = label_panel(denoised, f'wavelet denoised ({wavelet}, VisuShrink)')
    p3 = label_panel(residual, 'removed (noise estimate, 128=no change)')

    out_path = outdir / f'{stem}.png'
    cv2.imwrite(str(out_path), np.hstack([p1, p2, p3]))
    return f'{stem}: native {native_hw[1]}x{native_hw[0]}, wrote {out_path.name}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument('--max-dim', type=int, default=DEFAULT_MAX_DIM)
    ap.add_argument('--wavelet', default=DEFAULT_WAVELET,
                    help="any name from pywt.wavelist(), e.g. 'haar', 'db2', 'sym9'")
    ap.add_argument('--levels', type=int, default=DEFAULT_LEVELS)
    args = ap.parse_args()

    if args.outdir.exists():
        for f in args.outdir.glob('*.png'):
            f.unlink()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='wavelet_denoise_vips_') as tmp:
        tmpdir = Path(tmp)
        for stem, rel_path in CASES.items():
            print(process_case(stem, rel_path, args.outdir, args.max_dim,
                               args.wavelet, args.levels, tmpdir), flush=True)
    print(f'\nwrote previews to {args.outdir}')


if __name__ == '__main__':
    main()
