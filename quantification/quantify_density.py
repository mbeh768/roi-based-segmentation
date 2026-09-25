"""Relative green density by depth into myocardium, per algorithm.md.

    rho = sum(G) / (sum(G) + sum(R))    over band AND tissue

measured twice per slide: once with the depth bands grown from the manual
annotation, once with them grown from the model's held-out prediction. The two
arms differ ONLY in which mask the distance transform starts from -- same
image, same tissue mask -- so a difference between them is attributable to the
segmentation and nothing else.

Bands are exclusive rings measured outward from the mask boundary, so every
tissue pixel falls in exactly one of them:

    0-99 um     0 < d < 100         100-199 um  100 <= d < 200
    >200 um     d >= 200

The mask interior is d = 0 and therefore in no ring. A whole-foreground rho is
reported alongside as a diagnostic: it does not depend on the segmentation, so
it is identical in both arms by construction and is not a result.

Denoising is NOT applied yet -- algorithm.md's background subtraction step is
still to come, and every number here is on raw 8-bit intensities.

Usage:
    python quantification/quantify_density.py --global-only --qc-dir qc/
    python quantification/quantify_density.py --qc-dir qc/
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'train.py').exists())
sys.path.insert(0, str(ROOT))

from data_loader.data_loaders import case_id      # noqa: E402

# (label, inclusive lower bound, exclusive upper bound) in microns from the mask
# boundary; None is unbounded
BANDS = [('0-99 um', 0.0, 100.0), ('100-199 um', 100.0, 200.0), ('>200 um', 200.0, None)]
CLOSE_RADIUS = 8        # dilate-erode kernel radius, px
ISLAND_FRAC = 0.001     # drop components smaller than this share of the largest
CLIP_PCT = 95           # bright tail capped here before Otsu; see otsu_tissue_mask

QC_SIDE = 760           # long side of one QC panel, px
QC_TARGET_MEAN = 110.0  # QC renders only; measurement never sees this gain
BAND_COLOURS = ((250, 214, 96), (233, 138, 62), (198, 66, 66))
RINGS_NOTE = 'yellow 0-99, orange 100-199, red >200'


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def load_rgb(path):
    """(R, G, B) full-resolution uint8.

    Three TIFF layouts occur in this dataset -- interleaved samples, one page
    per channel, and three planes packed behind a single IFD -- and tifffile's
    series reader resolves all three, so only the axis order is handled here.
    uint16 is full-range, so /257 puts it on the same 0-255 scale and
    intensities stay comparable across slides.
    """
    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0]
        arr, axes = series.asarray(), series.axes
    planes = [arr[c] for c in range(3)] if axes.startswith('C') \
        else [arr[..., c] for c in range(3)]
    out = []
    for p in planes:
        if p.dtype == np.uint16:
            # rounds and saturates straight to uint8; going via float32 would
            # allocate a 5.7 GB temporary on the largest slide
            p = cv2.convertScaleAbs(p, alpha=1.0 / 257.0)
        elif p.dtype != np.uint8:
            raise ValueError(f'unexpected dtype {p.dtype} in {path}')
        out.append(np.ascontiguousarray(p))
    return out


def load_mask(path):
    """Annotation mask. Thresholded at the midpoint, not at >0: some source
    masks carry a low near-zero fill value alongside the real annotation."""
    arr = tifffile.imread(str(path))
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
    return arr >= 128


def find(raw_dir, stem, kind):
    hits = sorted(Path(raw_dir).glob(f'*/{stem}_{kind}.*'))
    if not hits:
        raise SystemExit(f'no {kind} for {stem} under {raw_dir}')
    return hits[0]


# ---------------------------------------------------------------------------
# Tissue and bands
# ---------------------------------------------------------------------------

def channel_mean(R, G, B):
    """Not the red channel: which fluorophore carries the tissue varies by
    slide, and red finds only the staining rim on 17P391_7A."""
    return ((R.astype(np.uint16) + G + B) // 3).astype(np.uint8)


def otsu_tissue_mask(image, scanned):
    """Otsu over `image`, foreground the brighter class. Unscanned canvas stays
    out of the histogram, and the bright tail is capped at CLIP_PCT first --
    a few saturated pixels otherwise let Otsu split everything against them,
    giving threshold 46 instead of 5 on 21PControl-2. The threshold applies to
    the unclipped image."""
    if not scanned.any():
        return np.zeros(image.shape, bool)
    vals = image[scanned]
    capped = np.minimum(vals, np.percentile(vals, CLIP_PCT)).astype(np.uint8)
    thresh, _ = cv2.threshold(capped.reshape(-1, 1), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    del vals, capped
    return scanned & (image > thresh)


def close_mask(mask, radius=CLOSE_RADIUS):
    """Dilate then erode: fills speckle without growing the tissue outline."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)


def drop_islands(mask, frac=ISLAND_FRAC):
    """Relabel components smaller than `frac` of the largest as background.
    Debris, bubbles and fluorescent dust all survive Otsu; none is myocardium.
    Returns the kept mask, what was removed, and how many components went."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask, np.zeros_like(mask), 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= frac * areas.max()) + 1
    lut = np.zeros(n, np.uint8)
    lut[keep] = 1
    kept = lut[labels].astype(bool)
    return kept, mask & ~kept, int(n - 1 - len(keep))


def rings(mask, um_per_px):
    """Exclusive depth rings outward from the mask boundary, in microns."""
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5) * um_per_px
    out = []
    for label, lo, hi in BANDS:
        band = dist > lo if lo == 0 else dist >= lo
        if hi is not None:
            band &= dist < hi
        out.append((label, band))
    return out


def rho(G, R, region):
    """sum(G) / (sum(G) + sum(R)) over a region, and the sums behind it."""
    n = int(np.count_nonzero(region))
    if n == 0:
        return float('nan'), 0, 0, 0
    g = int(G[region].sum(dtype=np.int64))
    r = int(R[region].sum(dtype=np.int64))
    return (g / (g + r) if (g + r) else float('nan')), g, r, n


# ---------------------------------------------------------------------------
# Quality assurance. Renders only -- the gain below exists so a section is
# visible on a page and would corrupt any measurement.
# ---------------------------------------------------------------------------

def shrink(mask, size):
    """Any foreground in a block survives. At ~35:1 a nearest sample would drop
    most small islands, and they are what the sheet has to show."""
    return cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_AREA) > 0


def qc_size(shape, long_side=QC_SIDE):
    H, W = shape
    s = long_side / max(H, W)
    return max(1, round(W * s)), max(1, round(H * s))


def qc_rgb(R, G, B, size):
    """A visible render at one global gain for all three channels. Per-channel
    would normalise away a channel imbalance, which on some slides is the defect
    the sheet exists to show."""
    planes = [cv2.resize(c, size, interpolation=cv2.INTER_AREA) for c in (R, G, B)]
    gain = QC_TARGET_MEAN / max(float(np.mean([p.mean() for p in planes])), 1e-6)
    return np.clip(np.stack(planes, -1).astype(np.float32) * gain, 0, 255).astype(np.uint8)


def tint(base, mask, colour, alpha=0.55):
    out = base.copy()
    out[mask] = (np.array(colour) * alpha + out[mask] * (1 - alpha)).astype(np.uint8)
    return out


def qc_sheet(path, stem, panels, note, cols=4):
    rows = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.2 * rows), squeeze=False)
    fig.patch.set_facecolor('#fcfcfb')
    flat = axes.ravel()
    for ax, (title, img) in zip(flat, panels):
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=8, color='#0b0b0b')
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in flat[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(f'{stem}\n{note}', fontsize=10, color='#52514e')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140, facecolor='#fcfcfb')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--raw-dir',
                    default=json.loads((ROOT / 'config.json').read_text())
                    ['preprocess']['raw_image_directory'])
    ap.add_argument('--metrics', default=str(ROOT / 'figures-for-pub' / 'baseline'
                                             / 'k-fold-baseline-2026-09-02_slide_metrics.json'))
    ap.add_argument('--voxel', default=str(HERE / 'microns_per_pixel.json'))
    ap.add_argument('--predictions', default=str(HERE / 'predictions'))
    ap.add_argument('--out', default=str(HERE / 'density.csv'))
    ap.add_argument('--qc-dir', default=None,
                    help='write a per-slide quality-assurance sheet there')
    ap.add_argument('--global-only', action='store_true',
                    help='tissue selection and whole-foreground rho only. Needs no '
                         'voxel size, so tissue selection can be checked before the '
                         'microns/pixel table is filled in.')
    ap.add_argument('--slides', nargs='*', default=None)
    a = ap.parse_args()

    voxel = {}
    if not a.global_only:
        if not Path(a.voxel).exists():
            raise SystemExit(f'{a.voxel} not found -- fill it in, or pass --global-only')
        voxel = json.loads(Path(a.voxel).read_text())

    stems = a.slides or [s for f in json.loads(Path(a.metrics).read_text())['folds']
                         for s in f['per_slide']]
    # up front, not per slide: the check is free and each slide costs minutes
    missing = [s for s in stems if not (voxel.get(s) or {}).get('um_per_px')]
    if voxel and missing:
        raise SystemExit(f'no um_per_px in {Path(a.voxel).name} for: ' + ', '.join(missing))

    qc = Path(a.qc_dir) if a.qc_dir else None
    if qc:
        qc.mkdir(parents=True, exist_ok=True)

    rows, summary = [], []
    for stem in stems:
        R, G, B = load_rgb(find(a.raw_dir, stem, 'image'))
        gt = load_mask(find(a.raw_dir, stem, 'mask'))
        if gt.shape != R.shape:
            raise SystemExit(f'{stem}: mask {gt.shape} != image {R.shape}')

        scanned = (R > 0) | (G > 0) | (B > 0)
        size = qc_size(R.shape)
        panels = []
        if qc:
            view = qc_rgb(R, G, B, size)
            panels.append(('raw', view))
            panels.append(('ground-truth annotation',
                           tint(view, shrink(gt, size), (80, 150, 255))))

        m = otsu_tissue_mask(channel_mean(R, G, B), scanned)
        if qc:
            panels.append(('1. Otsu on red', shrink(m, size)))
        m = close_mask(m)
        if qc:
            panels.append((f'2. dilate-erode r={CLOSE_RADIUS}', shrink(m, size)))
        tissue, removed, dropped = drop_islands(m)
        if qc:
            grey = np.stack([shrink(tissue, size).astype(np.uint8) * 255] * 3, -1)
            panels.append((f'3. islands dropped ({dropped}, red)',
                           tint(grey, shrink(removed, size), (230, 60, 60), 1.0)))
        del m

        g_rho, _, _, g_n = rho(G, R, tissue)
        gt_tissue = float((gt & tissue).sum() / max(1, gt.sum()))
        gt_unscanned = float((gt & ~scanned).sum() / max(1, gt.sum()))
        print(f'  {stem:<22} tissue {tissue.mean()*100:5.1f}%  -{dropped} islands   '
              f'gt tissue {gt_tissue*100:5.1f}%  gt unscanned {gt_unscanned*100:5.1f}%   '
              f'global rho {g_rho:.4f}')
        summary.append({'slide': stem, 'case': case_id(stem),
                        'tissue_frac': float(tissue.mean()), 'islands_dropped': dropped,
                        'gt_tissue_frac': gt_tissue, 'gt_unscanned_frac': gt_unscanned,
                        'global_rho': g_rho, 'global_px': g_n})

        if not a.global_only:
            um = (voxel.get(stem) or {}).get('um_per_px')
            if not um:
                raise SystemExit(f'{stem}: no um_per_px in {Path(a.voxel).name}')
            for arm in ('ground truth', 'predicted'):
                if arm == 'ground truth':
                    mask = gt
                else:
                    npz = Path(a.predictions) / f'{stem}.npz'
                    if not npz.exists():
                        raise SystemExit(f'{stem}: no prediction at {npz}')
                    mask = np.load(npz)['pred']
                    if mask.shape != R.shape:
                        raise SystemExit(f'{stem}: prediction {mask.shape} != {R.shape}')
                band_masks = rings(mask, float(um))
                for label, band in band_masks:
                    value, g, r, n = rho(G, R, band & tissue)
                    rows.append({'slide': stem, 'case': case_id(stem), 'arm': arm,
                                 'band': label, 'rho': value, 'g_sum': g, 'r_sum': r,
                                 'n_px': n, 'um_per_px': um})
                    print(f'      {arm:<13} {label:<11} rho {value:.4f}  n {n:>13,}')
                if qc:
                    canvas = np.zeros((size[1], size[0], 3), np.uint8)
                    for (_, band), colour in zip(band_masks, BAND_COLOURS):
                        canvas[shrink(band & tissue, size)] = colour
                    panels.append((f'rings, {arm}', canvas))
                del band_masks, mask

        if qc:
            qc_sheet(qc / f'{stem}.png', stem, panels,
                     f'tissue {tissue.mean()*100:.1f}% of frame   '
                     f'annotation {gt_tissue*100:.0f}% tissue, {gt_unscanned*100:.0f}% '
                     f'unscanned   global rho {g_rho:.4f}   ({RINGS_NOTE})')
            print(f'      qc -> {qc / f"{stem}.png"}')

        del R, G, B, gt, scanned, tissue, removed, panels

    with open(HERE / 'tissue_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f'\nwrote {HERE / "tissue_summary.csv"}  ({len(summary)} slides)')

    if rows:
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {a.out}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()
