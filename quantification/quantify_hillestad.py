"""
Quantification for Dr. Hillestad's manually-annotated (old-methodology) cases,
mirroring the steps used for pairs_quant_gt/ -- EXCEPT the 100um/200um cumulative
bands are pre-supplied here (100um.tif, 200um.tif, manually exported from the same
Photoshop canvas as MASK.tif -- see quantification_workspace/info.md) rather than
computed via our own distance transform. Region membership in each supplied TIFF is
"any non-white pixel" (solid color fill on a white background), verified by
inspection: MASK < 100um < 200um foreground area, i.e. the same cumulative-band
convention used elsewhere in this repo.

The continuous radial-distance intensity profile IS still computed here (same
signed-distance-transform approach as quantify_intensity.py), since that needs only
MASK.tif + microns/pixel, not the pre-supplied band files.

Per quantification_workspace/info.md: for this legacy data "RGB may only contain
meaningful data in red and green channels" -- blue is still reported but is flagged
as unreliable in plot titles.

MEMORY NOTES (these slides are up to 26618x32600 = 868 MPx; a naive implementation
peaks >20 GB and will take the machine down):
  * never materialize float64 copies of a full channel -- the radial profile is
    accumulated with np.bincount over row blocks instead
  * only one full-size float32 distance array is alive at a time (signed distance is
    built in-place), and it is downcast to int32 bins and freed before RGB is loaded
  * the RGBA output buffer is allocated once, in BGRA order, not copied
  * RLIMIT_AS is set (--mem-limit-gb) so that if an allocation does blow past the
    budget the process dies with MemoryError instead of swapping the box to death

Usage:
    python quantify_hillestad.py --case-dir <dir with RGB/MASK/100um/200um .tif> \
        --microns-per-pixel 2.5562774308232 --stem AJKY121_8 --outdir hillestad_quant_gt/AJKY121_8
"""
import argparse
import gc
import resource
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from quantify_bands import COLOR_SUBENDOCARDIUM, COLOR_100UM, COLOR_200UM

BAND_LABELS = ['subendocardium', '+100um', '+200um']
CH_COLORS = {'R': '#d62728', 'G': '#2ca02c', 'B': '#1f77b4'}
ROWS_PER_BLOCK = 2048          # ~250 MB of uint8 RGB per block at these widths


def set_mem_cap(gb):
    """Hard-cap this process's heap so a runaway allocation kills the process rather
    than the machine.

    RLIMIT_DATA, not RLIMIT_AS: RLIMIT_AS caps the *virtual* address space, which
    numpy/OpenBLAS inflate with large reservations they never fault in (a run whose
    real RSS peaked at 2 GB still tripped a 10 GB RLIMIT_AS). Since Linux 4.7
    RLIMIT_DATA covers anonymous mmap'd memory, so it tracks the allocations we
    actually care about.
    """
    if gb <= 0:
        return
    nbytes = int(gb * 1024 ** 3)
    resource.setrlimit(resource.RLIMIT_DATA, (nbytes, nbytes))
    print(f"memory cap: {gb:.1f} GB (RLIMIT_DATA)")


def nonwhite_mask(path):
    """Region membership = any non-white pixel. Built with one bool temporary at a
    time rather than np.all(arr == 255, axis=-1), which needs a full (H,W,3) bool."""
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:                       # already single-channel
        m = arr != 255
    else:
        m = arr[..., 0] != 255
        for c in range(1, arr.shape[2]):
            m |= arr[..., c] != 255
    del arr
    gc.collect()
    return m


def save_rgba(mask_bool, color_rgb, path):
    """Foreground-only RGBA PNG. Allocates the BGRA buffer once (cv2 wants BGRA)
    instead of building RGBA and then making a reordered copy."""
    h, w = mask_bool.shape
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    r, g, b = color_rgb
    bgra[..., 0][mask_bool] = b
    bgra[..., 1][mask_bool] = g
    bgra[..., 2][mask_bool] = r
    bgra[..., 3][mask_bool] = 255
    cv2.imwrite(str(path), bgra)
    del bgra
    gc.collect()
    print(f"saved -> {path}  ({int(mask_bool.sum())} px, {100*mask_bool.mean():.2f}% of frame)")


def band_means_blocked(band_masks, band_labels, rgb_path):
    """Mean R/G/B inside each band, accumulated over row blocks so the full RGB
    array is never combined with full-size float temporaries."""
    n = len(band_masks)
    sums = np.zeros((n, 3), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)

    with tifffile.TiffFile(str(rgb_path)) as tf:
        rgb = tf.asarray()                  # (H, W, 3) uint8 -- 2.4 GB at worst
    H = rgb.shape[0]
    for y0 in range(0, H, ROWS_PER_BLOCK):
        y1 = min(y0 + ROWS_PER_BLOCK, H)
        blk = rgb[y0:y1]                    # view, no copy
        for i, bm in enumerate(band_masks):
            sel = bm[y0:y1]
            k = int(sel.sum())
            if not k:
                continue
            counts[i] += k
            # blk[sel] is (k, 3) -- only as big as the selected pixels
            sums[i] += blk[sel].sum(axis=0, dtype=np.float64)
    del rgb
    gc.collect()

    rows = []
    for i, label in enumerate(band_labels):
        if counts[i]:
            r_m, g_m, b_m = (sums[i] / counts[i])
        else:
            r_m = g_m = b_m = float('nan')
        rows.append((label, float(r_m), float(g_m), float(b_m),
                     float(g_m / r_m) if r_m else float('nan'), int(counts[i])))
    return rows


def signed_distance_px(mask):
    """Signed Euclidean distance in PIXELS: negative inside the mask, positive
    outside. Only one full-size float32 array survives the call."""
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)   # >0 outside
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)    # >0 inside
    dist -= inside          # supports are disjoint, so this is exactly the signed field
    del inside
    gc.collect()
    return dist


def radial_profile_blocked(dist, lo, n_bins, rgb_path):
    """Accumulate per-distance-bin channel sums with np.bincount over row blocks.

    `dist` is the full-size float32 signed distance (in pixels). Bins are derived
    per row block rather than as a full-size int32 array: holding a (H,W) int32
    alongside the (H,W) float32 costs an extra 2.7 GB at full resolution and is what
    tipped this over the memory cap. Likewise the weights passed to bincount are one
    block tall, never a whole float64 channel (6.5 GB each).
    """
    counts = np.zeros(n_bins, dtype=np.int64)
    sums = np.zeros((3, n_bins), dtype=np.float64)

    with tifffile.TiffFile(str(rgb_path)) as tf:
        rgb = tf.asarray()
    H = rgb.shape[0]
    for y0 in range(0, H, ROWS_PER_BLOCK):
        y1 = min(y0 + ROWS_PER_BLOCK, H)
        b = (np.rint(dist[y0:y1]).astype(np.int32) - lo).ravel()   # block-sized only
        counts += np.bincount(b, minlength=n_bins)
        blk = rgb[y0:y1]
        for c in range(3):
            sums[c] += np.bincount(
                b, weights=blk[..., c].ravel().astype(np.float64, copy=False),
                minlength=n_bins,
            )
        del b
    del rgb
    gc.collect()
    return counts, sums


def save_bar_intensity(rows, stem, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(rows))
    w = 0.25
    for i, ch in enumerate(['R', 'G', 'B']):
        ax.bar(x + (i - 1) * w, [r[1 + i] for r in rows], width=w, label=ch, color=CH_COLORS[ch])
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel('mean pixel intensity (0-255)')
    ax.set_title(f'{stem}: intensity by band\n(B channel unreliable for this legacy data -- see info.md)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved -> {path}")


def save_bar_ratio(rows, stem, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(rows))
    ax.bar(x, [r[4] for r in rows], color=CH_COLORS['G'])
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel('mean-G / mean-R')
    ax.set_title(f'{stem}: green/red ratio by band')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved -> {path}")


def save_radial(dist_um, r_mean, g_mean, b_mean, g_over_r, stem, outdir):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(dist_um, r_mean, color=CH_COLORS['R'], label='R', linewidth=1)
    ax.plot(dist_um, g_mean, color=CH_COLORS['G'], label='G', linewidth=1)
    ax.plot(dist_um, b_mean, color=CH_COLORS['B'], label='B', linewidth=1)
    for edge in (0, 100, 200):
        ax.axvline(edge, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('signed distance from mask boundary (um)  [negative = inside mask]')
    ax.set_ylabel('mean pixel intensity (0-255)')
    ax.set_title(f'{stem}: radial intensity profile (1px resolution)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / 'radial_intensity.png', dpi=150)
    plt.close(fig)
    print(f"saved -> {outdir / 'radial_intensity.png'}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(dist_um, g_over_r, color=CH_COLORS['G'], linewidth=1)
    for edge in (0, 100, 200):
        ax.axvline(edge, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('signed distance from mask boundary (um)  [negative = inside mask]')
    ax.set_ylabel('mean-G / mean-R')
    ax.set_title(f'{stem}: radial green/red ratio profile (1px resolution)')
    fig.tight_layout()
    fig.savefig(outdir / 'radial_ratio.png', dpi=150)
    plt.close(fig)
    print(f"saved -> {outdir / 'radial_ratio.png'}")


def main(args):
    set_mem_cap(args.mem_limit_gb)

    case_dir = Path(args.case_dir)
    rgb_path = case_dir / "RGB.tif"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    um = args.microns_per_pixel

    # --- region masks (bool, ~0.8 GB each at worst) -------------------------------
    mask = nonwhite_mask(case_dir / "MASK.tif")
    band_100 = nonwhite_mask(case_dir / "100um.tif")
    band_200 = nonwhite_mask(case_dir / "200um.tif")
    print(f"{args.stem}: MASK={int(mask.sum())}px ({100*mask.mean():.2f}%)  "
          f"100um={int(band_100.sum())}px ({100*band_100.mean():.2f}%)  "
          f"200um={int(band_200.sum())}px ({100*band_200.mean():.2f}%)")
    if not (mask.sum() < band_100.sum() < band_200.sum()):
        print("WARNING: bands are not monotonically increasing -- check source files!")

    # --- band means (chunked over row blocks) -------------------------------------
    rows = band_means_blocked([mask, band_100, band_200], BAND_LABELS, rgb_path)
    with open(outdir / 'band_means.csv', 'w') as f:
        f.write('band,R_mean,G_mean,B_mean,G_over_R,n_px\n')
        for r in rows:
            f.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]}\n")
    print(f"saved -> {outdir / 'band_means.csv'}")
    save_bar_intensity(rows, args.stem, outdir / 'bar_intensity.png')
    save_bar_ratio(rows, args.stem, outdir / 'bar_ratio.png')

    # --- region PNGs (one full-size RGBA buffer alive at a time) ------------------
    save_rgba(mask, COLOR_SUBENDOCARDIUM, outdir / "subendocardium.png")
    save_rgba(band_100, COLOR_100UM, outdir / "100um.png")
    save_rgba(band_200, COLOR_200UM, outdir / "200um.png")
    del band_100, band_200
    gc.collect()

    # --- radial profile -----------------------------------------------------------
    # Only the float32 signed-distance field is kept full-size; bins are derived per
    # row block inside radial_profile_blocked. `mask` is dropped first -- it is only
    # needed to build the distance field.
    dist = signed_distance_px(mask)
    del mask
    gc.collect()

    lo = int(np.floor(dist.min()))
    hi = int(np.ceil(dist.max()))
    n_bins = hi - lo + 1
    counts, sums = radial_profile_blocked(dist, lo, n_bins, rgb_path)
    del dist
    gc.collect()

    valid = counts > 0
    dist_um = np.arange(lo, hi + 1)[valid] * um
    n_px = counts[valid]
    r_mean = sums[0][valid] / n_px
    g_mean = sums[1][valid] / n_px
    b_mean = sums[2][valid] / n_px
    g_over_r = np.divide(g_mean, r_mean, out=np.full_like(g_mean, np.nan), where=r_mean > 0)

    with open(outdir / 'radial_profile.csv', 'w') as f:
        f.write('distance_um,R_mean,G_mean,B_mean,G_over_R,n_px\n')
        for i in range(len(dist_um)):
            f.write(f"{dist_um[i]:.3f},{r_mean[i]:.4f},{g_mean[i]:.4f},{b_mean[i]:.4f},"
                    f"{g_over_r[i]:.4f},{n_px[i]}\n")
    print(f"saved -> {outdir / 'radial_profile.csv'}")
    save_radial(dist_um, r_mean, g_mean, b_mean, g_over_r, args.stem, outdir)

    print(f"\ndone -> {outdir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case-dir', required=True,
                        help='dir containing RGB.tif, MASK.tif, 100um.tif, 200um.tif')
    parser.add_argument('--microns-per-pixel', type=float, required=True)
    parser.add_argument('--stem', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--mem-limit-gb', type=float, default=12.0,
                        help='hard RLIMIT_AS cap; 0 disables (default: 12)')
    main(parser.parse_args())
