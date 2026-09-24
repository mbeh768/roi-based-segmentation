"""
Epicardium intensity-by-distance quantification pipeline.

Standalone script for collaborators: point it at a full-resolution RGB slide
and a binary mask TIFF, and it measures how R/G/B intensity varies with
distance from the mask boundary, across the tissue the mask does NOT cover.

WHAT IT DOES, IN ORDER:
  1. Prompts for the RGB image and the mask TIFF.
  2. Detects tissue automatically: Otsu threshold on the red channel (the
     cardiomyocyte stain gives the cleanest tissue/background split), then
     morphological closing (radius 15) to fill small speckled holes in that
     mask without touching real cavities/lumens, which are much larger than
     the closing kernel. This combination was validated by hand against 20
     slides before being written into this pipeline.
  3. Subtracts the user's mask from that tissue mask: epicardium region =
     (auto-detected tissue) MINUS (annotated region). This is the tissue
     the mask does not already account for.
  4. Computes, for every pixel in the epicardium region, its signed distance
     in pixels to the mask boundary (negative would mean inside the mask,
     but since the mask itself has just been subtracted out, in practice
     every epicardium pixel has a positive distance -- it is by
     construction outside the mask).
  5. Bins every epicardium pixel by its (integer-pixel) distance and
     averages R, G, and B intensity in each bin -- the same "radial
     profile" approach used during method development
     (quantify_intensity.py's radial_profile()), just restricted to the
     epicardium region here instead of the whole slide.
  6. Writes a CSV (distance_px, R_mean, G_mean, B_mean, n_px), a plot, and
     chosen_region.png -- the original image with the quantified region
     tinted pink, so it's a direct visual check of exactly which pixels
     the profile in the CSV came from.

WHY OTSU ON THE RED CHANNEL ALONE, NO EXTRA CLEANUP: earlier attempts in
development added percentile clipping and statistical noise-floor estimation
on top of Otsu to compensate for scanner artifacts and uneven staining. Those
made some slides worse. Plain Otsu on properly-imaged slides (excluding
completely-black scanner/stitching-gap pixels from the histogram) turned out
to be the most reliable option across the 20 slides tested. If a slide was
imaged poorly (very uneven illumination, near-zero signal over most of the
tissue), this pipeline will not compensate for that -- image slides
consistently and this threshold works well.

DEPENDENCIES: numpy, opencv-python (cv2), tifffile, matplotlib. No libvips.
tkinter (stdlib) is used for the file pickers when a display is available;
everywhere else this falls back to typed console paths.

Usage:
    python quantify_epicardium.py                 # opens file pickers
    python quantify_epicardium.py --image slide.tif --mask mask.tiff
(if --image/--mask are omitted and no display is available, it asks for
paths on the console instead)
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile

CLOSE_RADIUS = 15   # px; see epicardium_otsu_closed_preview.py -- fills speckle,
                     # leaves real cavities (much larger than this) untouched


# ---------------------------------------------------------------------------
# File prompts
#
# Same approach as segment.py in this folder: probe for a display up front
# (OpenCV/Tk both misbehave badly if you try to open a window with none
# available), then either open a native file picker or fall back to a typed
# console prompt.
# ---------------------------------------------------------------------------

def gui_available():
    """True if we can safely open a GUI window.

    Windows and macOS always have a window server available to a logged-in
    user; on Linux the session environment variables are the reliable
    signal. See segment.py's gui_available() for the same check.
    """
    if sys.platform in ('win32', 'darwin'):
        return True
    import os
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


def pick_file_dialog(title):
    """Open a native file picker. Returns a Path, or None if cancelled or
    unavailable (e.g. tkinter not installed)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[('TIFF images', '*.tif *.tiff *.TIF *.TIFF'), ('All files', '*.*')],
    )
    root.destroy()
    return Path(path) if path else None


def prompt_for_path(message):
    """Ask until the user gives a path to a file that actually exists.
    Quotes typed around a drag-and-dropped path (common on some terminals)
    are stripped automatically."""
    while True:
        raw = input(message).strip().strip('"').strip("'")
        path = Path(raw)
        if path.is_file():
            return path
        print(f'  "{raw}" is not a file -- try again (or Ctrl+C to quit).')


def resolve_input_path(cli_value, title, console_message):
    """--image/--mask if given, otherwise a file picker if a display is
    available, otherwise a typed console prompt."""
    if cli_value:
        path = Path(cli_value)
        if not path.is_file():
            sys.exit(f'ERROR: file not found: {path}')
        return path
    if gui_available():
        print(f'Opening file picker: {title}...')
        path = pick_file_dialog(title)
        if path is not None:
            return path
        print('  (no file picker available -- falling back to console)')
    return prompt_for_path(console_message)


# ---------------------------------------------------------------------------
# Reading the full-resolution RGB image
#
# These slides show up on disk in three different layouts (confirmed while
# developing this pipeline against the real dataset):
#   A. a standard interleaved RGB TIFF: one page, 3 samples per pixel.
#   B. an ImageJ-style stack of 3 separate single-band pages, in R,G,B page
#      order.
#   C. the same 3-channel ImageJ stack, but large enough that ImageJ wrote
#      it as a single classic-TIFF IFD with the 3 channel planes packed
#      contiguously right after it (raw, uncompressed), rather than as 3
#      separate IFDs -- there is no page to "seek to" for channels 1 and 2,
#      their bytes just start immediately after channel 0's.
# Each channel is read and (if needed) rescaled to uint8 ONE AT A TIME,
# never all three stacked into memory at once, to keep peak memory bounded
# on gigapixel slides regardless of source bit depth.
# ---------------------------------------------------------------------------

def rescale_to_uint8(arr):
    """uint8 passes through untouched. uint16 (full 0-65535 range, seen on
    some slides in this dataset) is rescaled to 0-255 so intensity values
    are comparable to the 8-bit slides. Done in float32, not float64, to
    keep the temporary copy this makes half the size."""
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return np.round(arr.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    raise ValueError(f'unexpected channel dtype {arr.dtype}')


def packed_plane_layout(path):
    """If this file is layout C (see module docstring), return
    (n_planes, byte_offsets, (H, W), numpy_dtype). Otherwise return None.
    Reads only the header/first-IFD metadata, never pixel data."""
    with tifffile.TiffFile(str(path)) as tf:
        if len(tf.pages) != 1:
            return None
        meta = tf.imagej_metadata
        if not meta or meta.get('images', 1) <= 1:
            return None
        page = tf.pages[0]
        n_planes = int(meta['images'])
        offset0, nbytes0 = page.dataoffsets[0], page.databytecounts[0]
        h, w = page.shape
        if nbytes0 != h * w * page.dtype.itemsize:
            return None  # not the simple one-strip-per-plane case this handles
        offsets = [offset0 + i * nbytes0 for i in range(n_planes)]
        return n_planes, offsets, (h, w), page.dtype


def read_packed_plane(path, offset, shape, dtype):
    """Read one raw, uncompressed plane directly from its byte offset --
    used for layout C, where tifffile/vips cannot address planes 1 and 2 as
    separate pages."""
    count = shape[0] * shape[1]
    with open(path, 'rb') as f:
        f.seek(offset)
        flat = np.fromfile(f, dtype=dtype, count=count)
    return flat.reshape(shape)


def load_rgb_fullres(path):
    """Return (R, G, B) as separate full-resolution uint8 (H, W) arrays.
    Kept separate rather than stacked into one (H, W, 3) array so that
    nothing here ever needs 3x a full channel's worth of memory at once."""
    with tifffile.TiffFile(str(path)) as tf:
        page = tf.pages[0]
        n_pages = len(tf.pages)
        bands = page.samplesperpixel

    if bands == 3 and n_pages == 1:
        # Layout A: one page, 3 interleaved samples per pixel.
        with tifffile.TiffFile(str(path)) as tf:
            arr = tf.pages[0].asarray()  # (H, W, 3)
        r = rescale_to_uint8(np.ascontiguousarray(arr[..., 0]))
        g = rescale_to_uint8(np.ascontiguousarray(arr[..., 1]))
        b = rescale_to_uint8(np.ascontiguousarray(arr[..., 2]))
        return r, g, b

    if bands == 1 and n_pages >= 3:
        # Layout B: 3 separate single-band pages, R,G,B page order.
        with tifffile.TiffFile(str(path)) as tf:
            r = rescale_to_uint8(tf.pages[0].asarray())
            g = rescale_to_uint8(tf.pages[1].asarray())
            b = rescale_to_uint8(tf.pages[2].asarray())
        return r, g, b

    packed = packed_plane_layout(path) if (bands == 1 and n_pages == 1) else None
    if packed is not None:
        # Layout C: 3 planes packed behind a single IFD -- address by hand.
        n_planes, offsets, shape, dtype = packed
        r = rescale_to_uint8(read_packed_plane(path, offsets[0], shape, dtype))
        g = rescale_to_uint8(read_packed_plane(path, offsets[1], shape, dtype))
        b = rescale_to_uint8(read_packed_plane(path, offsets[2], shape, dtype))
        return r, g, b

    raise ValueError(f'unrecognized TIFF layout for {path}: '
                     f'samplesperpixel={bands}, pages={n_pages}')


def load_mask(path):
    """Ground-truth annotation mask: foreground = pixel value >= 128 (some
    source masks carry a low near-zero fill value alongside the real
    annotation, so this thresholds at the midpoint rather than treating any
    nonzero value as foreground)."""
    arr = tifffile.imread(str(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr >= 128


# ---------------------------------------------------------------------------
# Tissue detection: Otsu + morphological closing
# ---------------------------------------------------------------------------

def otsu_tissue_mask(R, G, B):
    """Global Otsu threshold on the red channel, foreground = the brighter
    class. Completely black pixels (R=G=B=0 exactly) are excluded from the
    histogram first -- these are scanner/stitching artifacts (gaps between
    scan tiles, unscanned borders), not real background, and left in they
    distort the threshold."""
    real = (R > 0) | (G > 0) | (B > 0)
    if not real.any():
        return np.zeros(R.shape, dtype=bool)
    thresh, _ = cv2.threshold(R[real].reshape(-1, 1), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return real & (R > thresh)


def close_mask(mask, radius=CLOSE_RADIUS):
    """Dilate then erode: fills small speckled holes/gaps in the mask
    without meaningfully touching real cavities much larger than the
    kernel, and without growing the overall tissue outline."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k)
    return closed.astype(bool)


# ---------------------------------------------------------------------------
# Radial (distance-from-mask) intensity profile
# ---------------------------------------------------------------------------

def signed_distance_px(user_mask):
    """Per-pixel distance to the mask boundary, in pixels: positive outside
    the mask, negative inside."""
    dist_out = cv2.distanceTransform((~user_mask).astype(np.uint8), cv2.DIST_L2, 5)
    dist_in = cv2.distanceTransform(user_mask.astype(np.uint8), cv2.DIST_L2, 5)
    return np.where(user_mask, -dist_in, dist_out)


def radial_profile(signed_dist, region, R, G, B):
    """Mean R/G/B per integer-pixel distance bin, computed over `region`
    pixels only (everything outside `region` is excluded entirely, not
    just zeroed -- a zeroed pixel would still pull its bin's mean down)."""
    dist_region = signed_dist[region]
    bin_idx = np.round(dist_region).astype(np.int32)
    lo, hi = int(bin_idx.min()), int(bin_idx.max())
    n_bins = hi - lo + 1
    flat = bin_idx - lo

    Rr = R[region].astype(np.float64)
    Gr = G[region].astype(np.float64)
    Br = B[region].astype(np.float64)

    counts = np.bincount(flat, minlength=n_bins)
    r_sum = np.bincount(flat, weights=Rr, minlength=n_bins)
    g_sum = np.bincount(flat, weights=Gr, minlength=n_bins)
    b_sum = np.bincount(flat, weights=Br, minlength=n_bins)

    valid = counts > 0
    dist_px = np.arange(lo, hi + 1)[valid]
    r_mean = r_sum[valid] / counts[valid]
    g_mean = g_sum[valid] / counts[valid]
    b_mean = b_sum[valid] / counts[valid]
    return dist_px, r_mean, g_mean, b_mean, counts[valid]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_csv(path, dist_px, r_mean, g_mean, b_mean, n_px):
    with open(path, 'w') as f:
        f.write('distance_px,R_mean,G_mean,B_mean,n_px\n')
        for i in range(len(dist_px)):
            f.write(f'{dist_px[i]},{r_mean[i]:.4f},{g_mean[i]:.4f},{b_mean[i]:.4f},{n_px[i]}\n')
    print(f'saved -> {path}')


def save_plot(path, dist_px, r_mean, g_mean, b_mean, stem):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(dist_px, r_mean, color='#d62728', label='R', linewidth=1)
    ax.plot(dist_px, g_mean, color='#2ca02c', label='G', linewidth=1)
    ax.plot(dist_px, b_mean, color='#1f77b4', label='B', linewidth=1)
    ax.set_xlabel('distance from mask boundary (px)')
    ax.set_ylabel('mean pixel intensity (0-255)')
    ax.set_title(f'{stem}: epicardium intensity vs. distance from mask')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'saved -> {path}')


REGION_PREVIEW_MAX_DIM = 2000  # px; a sanity-check image, not data -- downsampled so it's
                               # actually fast to write and open at gigapixel source sizes
REGION_COLOR = (255, 0, 255)   # BGR magenta/pink -- same as TISSUE_COLOR in
                               # quant/epicardium_otsu_preview.py
REGION_ALPHA = 0.45            # same blend weight as that script's tint()


def save_region_image(path, R, G, B, region, max_dim=REGION_PREVIEW_MAX_DIM,
                      color=REGION_COLOR, alpha=REGION_ALPHA):
    """Original image with `region` tinted pink -- a direct visual check of
    exactly which pixels the radial profile measured (region here is the
    epicardium mask: closed Otsu tissue minus the user's mask). Same pink
    overlay (color + alpha blend) as quant/epicardium_otsu_preview.py's
    tint(), so this looks like the previews used to validate the method.
    Downsampled to max_dim on the long side before saving; this is a
    sanity-check image, not the data itself (that's the CSV)."""
    bgr = np.stack([B, G, R], axis=-1)  # cv2 uses BGR

    h, w = bgr.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        bgr = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_AREA)
        region = cv2.resize(region.astype(np.uint8), (bgr.shape[1], bgr.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)

    if region.any():
        patch = bgr[region].astype(np.float32)
        bgr[region] = np.clip(patch * (1 - alpha) + np.array(color, np.float32) * alpha,
                              0, 255).astype(np.uint8)

    cv2.imwrite(str(path), bgr)
    print(f'saved -> {path}  (downsampled to {bgr.shape[1]}x{bgr.shape[0]} for viewing)')


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Quantify epicardium intensity by distance from a mask.')
    ap.add_argument('--image', default=None,
                    help='full-resolution RGB slide TIFF. If omitted, a file picker opens.')
    ap.add_argument('--mask', default=None,
                    help='mask TIFF. If omitted, a file picker opens.')
    args = ap.parse_args()

    print('Epicardium intensity-by-distance quantification\n')
    image_path = resolve_input_path(
        args.image, 'Select the full-resolution RGB slide image',
        'Full-resolution RGB image (.tif/.tiff): ')
    mask_path = resolve_input_path(
        args.mask, 'Select the mask TIFF',
        'Mask TIFF (.tif/.tiff): ')

    print('\nreading image...')
    R, G, B = load_rgb_fullres(image_path)
    print(f'  image size: {R.shape[1]} x {R.shape[0]}')

    print('reading mask...')
    user_mask = load_mask(mask_path)
    if user_mask.shape != R.shape:
        sys.exit(f'ERROR: mask shape {user_mask.shape} != image shape {R.shape} -- '
                 f'these must be the same slide at the same resolution.')
    print(f'  mask foreground: {int(user_mask.sum()):,}px ({100 * user_mask.mean():.2f}%)')

    print('detecting tissue (otsu on red channel)...')
    tissue = otsu_tissue_mask(R, G, B)
    print(f'  raw otsu tissue: {int(tissue.sum()):,}px ({100 * tissue.mean():.2f}%)')

    print(f'closing tissue mask (radius={CLOSE_RADIUS})...')
    tissue_closed = close_mask(tissue)
    print(f'  closed tissue:   {int(tissue_closed.sum()):,}px ({100 * tissue_closed.mean():.2f}%)')

    epicardium = tissue_closed & ~user_mask
    print(f'epicardium region (closed tissue minus mask): '
          f'{int(epicardium.sum()):,}px ({100 * epicardium.mean():.2f}%)')

    print('computing distance-from-mask intensity profile...')
    signed_dist = signed_distance_px(user_mask)
    dist_px, r_mean, g_mean, b_mean, n_px = radial_profile(signed_dist, epicardium, R, G, B)

    stem = image_path.stem
    outdir = image_path.parent
    save_csv(outdir / f'{stem}_epicardium_profile.csv', dist_px, r_mean, g_mean, b_mean, n_px)
    save_plot(outdir / f'{stem}_epicardium_profile.png', dist_px, r_mean, g_mean, b_mean, stem)
    save_region_image(outdir / 'chosen_region.png', R, G, B, epicardium)

    print('\ndone.')


if __name__ == '__main__':
    main()
