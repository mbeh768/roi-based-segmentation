"""
Quantify RGB fluorescence intensity by distance band, using a ground-truth mask and
the full-resolution RGB TIFF from "RGB FULL - dataset" (confirmed via ImageJ LUT
metadata: page0=Red, page1=Green, page2=Blue -- NOT the raw .czi, whose 3 fluorophore
channels don't have a reliable fixed mapping to R/G/B).

Produces four plots + two CSVs per case:
  bar_intensity.png   -- grouped bar: mean R/G/B per cumulative band
                          (subendocardium, +100um, +200um -- each a superset of the last)
  bar_ratio.png        -- bar: mean-G / mean-R per cumulative band
  radial_intensity.png -- continuous curve: mean R/G/B vs signed distance (um) from the
                          mask boundary, binned at 1-pixel resolution (negative = inside
                          the mask, 0 = boundary, positive = outward into surrounding tissue)
  radial_ratio.png     -- continuous curve: mean-G / mean-R vs signed distance (um)
  band_means.csv        -- band, R_mean, G_mean, B_mean, G_over_R, n_px
  radial_profile.csv    -- distance_um, R_mean, G_mean, B_mean, G_over_R, n_px

Usage:
    python quantify_intensity.py --mask <mask.tif> --rgb <RGB_FULL_image.tiff> \
        --czi <source.czi> --outdir out/ --stem AJKV056-10-1-of-2
    python quantify_intensity.py --mask <mask.tif> --rgb <image.tiff> \
        --microns-per-pixel 2.5562774308232 --outdir out/
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from pipeline import czi_native_dims_and_scale

BAND_LABELS = ['subendocardium', '+100um', '+200um']
BAND_COLORS = {'R': '#d62728', 'G': '#2ca02c', 'B': '#1f77b4'}


def load_mask(path):
    arr = tifffile.imread(str(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr >= 128


def load_rgb(path):
    """Return (R, G, B) uint8 (H, W) arrays + a bit-depth label for plot titles.
    Confirmed page order via ImageJ LUT tags. Some slides in this dataset were
    exported as full-range 16-bit (0-65535) RGB TIFFs rather than 8-bit -- those
    are rescaled to 0-255 here (in memory only; the source file on disk is never
    touched) so mean-intensity values are comparable across slides."""
    arr = tifffile.imread(str(path))   # (3, H, W), page0=R page1=G page2=B
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"expected a 3-page (3,H,W) RGB TIFF, got shape {arr.shape}")
    if arr.dtype == np.uint8:
        bit_label = '8bit'
    elif arr.dtype == np.uint16:
        arr = np.round(arr.astype(np.float64) / 257.0).clip(0, 255).astype(np.uint8)
        bit_label = '16bit rescaled to 8bit'
    else:
        raise ValueError(f"unexpected RGB dtype {arr.dtype} for {path}")
    return arr[0], arr[1], arr[2], bit_label


def resolve_microns_per_pixel(args):
    if args.microns_per_pixel is not None:
        print(f"microns/pixel: {args.microns_per_pixel:.6f}  (CLI override)")
        return float(args.microns_per_pixel)
    if args.czi is None:
        raise ValueError("pass --microns-per-pixel or --czi to establish physical scale.")
    _, _, um_x, um_y = czi_native_dims_and_scale(args.czi)
    if um_x is None:
        raise ValueError(f"could not read microns/pixel from {args.czi}; use --microns-per-pixel.")
    if um_y is not None and abs(um_x - um_y) / um_x > 0.01:
        print(f"WARNING: anisotropic scaling (X={um_x:.4f}, Y={um_y:.4f} um/px); using X.")
    print(f"microns/pixel: {um_x:.6f}  (read from {Path(args.czi).name})")
    return float(um_x)


def band_means(mask, dist_out_um, R, G, B):
    """Cumulative bands: mask itself, mask+100um, mask+200um (each a superset of the last)."""
    band_masks = [
        mask,
        mask | (dist_out_um <= 100.0),
        mask | (dist_out_um <= 200.0),
    ]
    rows = []
    for label, bm in zip(BAND_LABELS, band_masks):
        n = int(bm.sum())
        r_mean = float(R[bm].mean()) if n else float('nan')
        g_mean = float(G[bm].mean()) if n else float('nan')
        b_mean = float(B[bm].mean()) if n else float('nan')
        rows.append((label, r_mean, g_mean, b_mean, g_mean / r_mean if r_mean else float('nan'), n))
    return rows


def radial_profile(signed_dist_um, um_per_px, R, G, B):
    """Mean R/G/B binned at 1-physical-pixel resolution of signed distance from the
    mask boundary (negative = inside the mask, positive = outside)."""
    bin_idx = np.round(signed_dist_um / um_per_px).astype(np.int32)
    lo, hi = int(bin_idx.min()), int(bin_idx.max())
    n_bins = hi - lo + 1
    flat = (bin_idx - lo).ravel()

    counts = np.bincount(flat, minlength=n_bins)
    r_sum = np.bincount(flat, weights=R.ravel().astype(np.float64), minlength=n_bins)
    g_sum = np.bincount(flat, weights=G.ravel().astype(np.float64), minlength=n_bins)
    b_sum = np.bincount(flat, weights=B.ravel().astype(np.float64), minlength=n_bins)

    valid = counts > 0
    dist_um = (np.arange(lo, hi + 1)[valid]) * um_per_px
    r_mean = r_sum[valid] / counts[valid]
    g_mean = g_sum[valid] / counts[valid]
    b_mean = b_sum[valid] / counts[valid]
    n_px = counts[valid]
    g_over_r = np.divide(g_mean, r_mean, out=np.full_like(g_mean, np.nan), where=r_mean > 0)
    return dist_um, r_mean, g_mean, b_mean, g_over_r, n_px


def _titled(stem, bit_label, suffix):
    base = f'{stem}: {suffix}'
    return f'{base}\n[{bit_label}]' if bit_label else base


def save_bar_intensity(rows, stem, path, bit_label=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(rows))
    w = 0.25
    for i, ch in enumerate(['R', 'G', 'B']):
        vals = [r[1 + i] for r in rows]
        ax.bar(x + (i - 1) * w, vals, width=w, label=ch, color=BAND_COLORS[ch])
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel('mean pixel intensity (0-255)')
    ax.set_title(_titled(stem, bit_label, 'intensity by band'), fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved -> {path}")


def save_bar_ratio(rows, stem, path, bit_label=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(rows))
    vals = [r[4] for r in rows]
    ax.bar(x, vals, color='#2ca02c')
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel('mean-G / mean-R')
    ax.set_title(_titled(stem, bit_label, 'green/red ratio by band'), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved -> {path}")


def save_radial_intensity(dist_um, r_mean, g_mean, b_mean, stem, path, bit_label=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(dist_um, r_mean, color=BAND_COLORS['R'], label='R', linewidth=1)
    ax.plot(dist_um, g_mean, color=BAND_COLORS['G'], label='G', linewidth=1)
    ax.plot(dist_um, b_mean, color=BAND_COLORS['B'], label='B', linewidth=1)
    for edge in (0, 100, 200):
        ax.axvline(edge, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('signed distance from mask boundary (um)  [negative = inside mask]')
    ax.set_ylabel('mean pixel intensity (0-255)')
    ax.set_title(_titled(stem, bit_label, 'radial intensity profile (1px resolution)'), fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved -> {path}")


def save_radial_ratio(dist_um, g_over_r, stem, path, bit_label=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(dist_um, g_over_r, color='#2ca02c', linewidth=1)
    for edge in (0, 100, 200):
        ax.axvline(edge, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('signed distance from mask boundary (um)  [negative = inside mask]')
    ax.set_ylabel('mean-G / mean-R')
    ax.set_title(_titled(stem, bit_label, 'radial green/red ratio profile (1px resolution)'), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved -> {path}")


def main(args):
    mask = load_mask(args.mask)
    R, G, B, bit_label = load_rgb(args.rgb)
    if R.shape != mask.shape:
        raise ValueError(f"mask shape {mask.shape} != RGB shape {R.shape} -- "
                         f"must be the same slide at the same resolution.")
    print(f"mask: {args.mask}  shape={mask.shape}  foreground={int(mask.sum())}px "
          f"({100*mask.mean():.2f}%)")

    um_per_px = resolve_microns_per_pixel(args)

    dist_out_px = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    dist_in_px = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    dist_out_um = dist_out_px * um_per_px
    signed_dist_um = np.where(mask, -dist_in_px * um_per_px, dist_out_um)

    stem = args.stem or Path(args.rgb).stem

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'bit_depth.txt').write_text(bit_label + '\n')
    print(f"RGB source bit depth: {bit_label}")

    rows = band_means(mask, dist_out_um, R, G, B)
    with open(outdir / 'band_means.csv', 'w') as f:
        f.write('band,R_mean,G_mean,B_mean,G_over_R,n_px\n')
        for r in rows:
            f.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]}\n")
    print(f"saved -> {outdir / 'band_means.csv'}")
    save_bar_intensity(rows, stem, outdir / 'bar_intensity.png', bit_label)
    save_bar_ratio(rows, stem, outdir / 'bar_ratio.png', bit_label)

    dist_um, r_mean, g_mean, b_mean, g_over_r, n_px = radial_profile(
        signed_dist_um, um_per_px, R, G, B)
    with open(outdir / 'radial_profile.csv', 'w') as f:
        f.write('distance_um,R_mean,G_mean,B_mean,G_over_R,n_px\n')
        for i in range(len(dist_um)):
            f.write(f"{dist_um[i]:.3f},{r_mean[i]:.4f},{g_mean[i]:.4f},{b_mean[i]:.4f},"
                    f"{g_over_r[i]:.4f},{n_px[i]}\n")
    print(f"saved -> {outdir / 'radial_profile.csv'}")
    save_radial_intensity(dist_um, r_mean, g_mean, b_mean, stem, outdir / 'radial_intensity.png', bit_label)
    save_radial_ratio(dist_um, g_over_r, stem, outdir / 'radial_ratio.png', bit_label)

    print(f"\ndone -> {outdir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mask', required=True, help='ground-truth mask (.tif/.tiff)')
    parser.add_argument('--rgb', required=True,
                        help='full-resolution RGB TIFF (from "RGB FULL - dataset", '
                             'NOT the raw .czi -- see module docstring)')
    parser.add_argument('--czi', default=None, help='source .czi to auto-read microns/pixel from')
    parser.add_argument('--microns-per-pixel', type=float, default=None,
                        help='override/skip auto voxel-size lookup entirely')
    parser.add_argument('--stem', default=None, help='label for plot titles (default: from --rgb)')
    parser.add_argument('--outdir', required=True)
    main(parser.parse_args())
