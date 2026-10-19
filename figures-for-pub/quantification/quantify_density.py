"""Relative MYL4 density by depth into myocardium, per algorithm.md.

    rho_px = mean over pixels of  g / (g + scale * r)   over band AND tissue

G is the MYL4 channel, R the counterstain, and `scale` is sum(G)/sum(R) over
the tissue outside the mask, which cancels the per-slide channel gain. The mask
is kept out of that reference because it holds the cavity.

One pixel, one vote. The ratio of sums this replaced was intensity-weighted, so
a handful of bright pixels could carry a band, and its >200 um value was pinned
near 0.5 by construction -- that band holds nearly every pixel the normaliser
is built from.

`scale` stays inside the per-pixel ratio: unlike a ratio of sums, a mean of
ratios is NOT invariant to a per-channel gain. rho_px_raw is the same measure
without it, carried so that cost is visible rather than assumed.

Bands are exclusive rings outward from the mask boundary -- 0-99, 100-199, 200+
um. The interior is at zero distance, so it falls in no ring.

ARMS differ ONLY in the mask those rings grow from. The image, the Otsu
threshold and the tissue mask are per-slide rather than per-arm, so they are
computed once and shared: adding an arm costs a distance transform, not another
pass over a gigapixel TIFF.

algorithm.md's background subtraction is not implemented; these are raw 8-bit
intensities, so each slide carries whatever zero point its scan had. See
EXCLUSIONS.md.

Usage:
    python figures-for-pub/quantification/quantify_density.py --global-only --qc-dir qc/
    python figures-for-pub/quantification/quantify_density.py --qc-dir qc/
    python figures-for-pub/quantification/quantify_density.py --arms "ground truth" SAM
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
import torch

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'train.py').exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'utils'))

from data_loader.data_loaders import case_id      # noqa: E402
from slide_level_metrics import to_native         # noqa: E402

# (label, inclusive lower bound, exclusive upper bound) in microns from the mask
# boundary; None is unbounded
BANDS = [('0-99 um', 0.0, 100.0), ('100-199 um', 100.0, 200.0), ('>200 um', 200.0, None)]
CLOSE_RADIUS = 8        # dilate-erode kernel radius, px
ISLAND_FRAC = 0.001     # tissue: drop components under this share of the largest
PRED_ISLAND_FRAC = 0.01  # same for a segmentation, which is one or two objects
CLIP_PCT = 95           # bright tail capped here before Otsu; see otsu_tissue_mask

QC_SIDE = 760           # long side of one QC panel, px
QC_TARGET_MEAN = 110.0  # QC renders only; measurement never sees this gain
BAND_COLOURS = ((250, 214, 96), (233, 138, 62), (198, 66, 66))
RINGS_NOTE = 'yellow 0-99, orange 100-199, red >200'


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def load_rgb(path):
    """(R, G, B) full-resolution uint8. tifffile's series reader resolves all
    three TIFF layouts in this dataset, so only the axis order is handled here;
    uint16 is full-range, so /257 keeps slides comparable."""
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
    """Otsu over `image`, brighter class is foreground. Unscanned canvas stays
    out of the histogram, and the bright tail is capped at CLIP_PCT first: a few
    saturated pixels otherwise drag the threshold to 46 instead of 5. The
    threshold applies to the unclipped image."""
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


def drop_islands(mask, frac=ISLAND_FRAC, with_removed=True):
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
    del labels, lut
    removed = (mask & ~kept) if with_removed else np.zeros((1, 1), bool)
    return kept, removed, int(n - 1 - len(keep))


def rings(mask, um_per_px, tissue):
    """Exclusive depth rings outward from the mask boundary, each already
    narrowed to tissue. Narrowed here, once: three call sites want the same
    region and every `band & tissue` is a full-frame bool array."""
    # thresholds converted to pixels rather than the field to microns: scaling
    # the field would allocate a second full-size float32
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    out = []
    for label, lo, hi in BANDS:
        band = dist > 0 if lo == 0 else dist >= lo / um_per_px
        if hi is not None:
            band &= dist < hi / um_per_px
        band &= tissue
        out.append((label, band))
    return out


def channel_sums(G, R, region):
    """Total MYL4, total counterstain, and pixel count over a region.

    Only the sums: the ratio-of-sums rho they used to form is gone. It was
    intensity-weighted, so a few bright pixels decided a band, and its >200 um
    value was pinned near 0.5 by construction. What the sums are still for is
    `scale`, which is a ratio of totals by definition.
    """
    n = int(np.count_nonzero(region))
    if n == 0:
        return 0, 0, 0
    return (int(G[region].sum(dtype=np.int64)),
            int(R[region].sum(dtype=np.int64)), n)


def rho_per_pixel(G, R, region, scale=1.0, rows=512):
    """Mean of per-pixel g / (g + scale*r), one vote per pixel.

    Also returned at scale 1.0, since a per-pixel ratio -- unlike a ratio of
    sums -- is not invariant to the per-channel gain.

    Row blocks because the >200 um region reaches ~250M px and a float32 copy
    does not fit beside the three channels. g = r = 0 has no ratio and is
    counted, not imputed.
    """
    total = total_raw = 0.0
    n = undefined = 0
    for y in range(0, region.shape[0], rows):
        sl = slice(y, y + rows)
        m = region[sl]
        if not m.any():
            continue
        g = G[sl][m].astype(np.float32)
        r = R[sl][m].astype(np.float32)
        d, d_raw = g + scale * r, g + r
        ok = d > 0
        total += float(np.divide(g[ok], d[ok]).sum(dtype=np.float64))
        total_raw += float(np.divide(g[ok], d_raw[ok]).sum(dtype=np.float64))
        n += int(np.count_nonzero(ok))
        undefined += int(ok.size - np.count_nonzero(ok))
        del g, r, d, d_raw, ok
    if n == 0:
        return float('nan'), float('nan'), 0, undefined
    return total / n, total_raw / n, n, undefined


def measure(G, R, band_masks):
    """rho per band, and the `scale` the bands share.

    scale is built from the bands TOGETHER, which is the tissue outside the
    mask: the mask holds the cavity, and averaging that in would put background
    into the reference.
    """
    sums = [channel_sums(G, R, band) for _, band in band_masks]
    gf, rf = (sum(s[i] for s in sums) for i in (0, 1))
    scale = gf / rf if rf else 1.0
    out = []
    for (label, band), (g, r, n) in zip(band_masks, sums):
        px, px_raw, _, undefined = rho_per_pixel(G, R, band, scale)
        out.append({'band': label, 'rho_px': px, 'rho_px_raw': px_raw,
                    'g_sum': g, 'r_sum': r, 'n_px': n,
                    'n_undefined': undefined, 'scale': scale})
    return out


# ---------------------------------------------------------------------------
# Arms. Each returns a native-resolution mask for the rings to grow from.
# Every arm but the annotation gets the same island filter, so a difference
# between two arms cannot come from post-processing.
# ---------------------------------------------------------------------------

def _npz(directory, stem):
    f = Path(directory) / f'{stem}.npz'
    if not f.exists():
        raise SystemExit(f'{stem}: no mask at {f}')
    return np.load(f)['pred']


ARMS = {
    'ground truth': lambda a, stem, ctx: load_mask(find(a.raw_dir, stem, 'mask')),
    'predicted':    lambda a, stem, ctx: _npz(a.predictions, stem),
    'SAM':          lambda a, stem, ctx: _npz(Path(a.sam_dir) / 'sam', stem),
    'microSAM':     lambda a, stem, ctx: _npz(Path(a.sam_dir) / 'microsam', stem),
    'ROI - tissue': lambda a, stem, ctx: to_native(
        torch.load(Path(a.data_dir) / 'images' / stem / f'roi_{stem}.pt',
                   weights_only=True).squeeze(0).numpy() > 0.5, *ctx['shape']
    ) & ~ctx['tissue'],
}


def arm_mask(name, a, stem, ctx):
    mask = ARMS[name](a, stem, ctx)
    if mask.shape != ctx['shape']:
        raise SystemExit(f'{stem}/{name}: {mask.shape} != {ctx["shape"]}')
    return mask if name == 'ground truth' \
        else drop_islands(mask, PRED_ISLAND_FRAC, False)[0]


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
    cfg = json.loads((ROOT / 'config.json').read_text())
    ap.add_argument('--raw-dir', default=cfg['preprocess']['raw_image_directory'])
    ap.add_argument('--data-dir', default=str(ROOT / cfg['data_loader']['args']['data_dir']),
                    help='holds roi_<stem>.pt, for the ROI - tissue arm')
    ap.add_argument('--metrics', default=str(ROOT / 'figures-for-pub' / 'model-baseline'
                                             / 'k-fold-bnaug-2026-10-17_slide_metrics.json'),
                    help='names the slides to measure')
    ap.add_argument('--voxel', default=str(HERE / 'microns_per_pixel.json'))
    ap.add_argument('--predictions', default=str(HERE / 'predictions'))
    ap.add_argument('--sam-dir', default=str(HERE / 'masks'),
                    help='written by export_sam_masks.py')
    ap.add_argument('--out', default=str(HERE / 'density.csv'))
    ap.add_argument('--arms', nargs='*', default=['ground truth', 'predicted'],
                    choices=list(ARMS), metavar='ARM',
                    help=f'any of: {", ".join(ARMS)}')
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
            panels.append(('1. Otsu on channel mean', shrink(m, size)))
        m = close_mask(m)
        if qc:
            panels.append((f'2. dilate-erode r={CLOSE_RADIUS}', shrink(m, size)))
        tissue, removed, dropped = drop_islands(m)
        if qc:
            grey = np.stack([shrink(tissue, size).astype(np.uint8) * 255] * 3, -1)
            panels.append((f'3. islands dropped ({dropped}, red)',
                           tint(grey, shrink(removed, size), (230, 60, 60), 1.0)))
        del m

        g_rho, _, _, g_n = rho_per_pixel(G, R, tissue)
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
            ctx = {'shape': R.shape, 'tissue': tissue}
            for arm in a.arms:
                band_masks = rings(arm_mask(arm, a, stem, ctx), float(um), tissue)
                for rec in measure(G, R, band_masks):
                    rows.append({'slide': stem, 'case': case_id(stem), 'arm': arm,
                                 **rec, 'um_per_px': um})
                    print(f'      {arm:<13} {rec["band"]:<11} '
                          f'rho_px {rec["rho_px"]:.4f}  n {rec["n_px"]:>13,}')
                if qc:
                    canvas = np.zeros((size[1], size[0], 3), np.uint8)
                    for (_, band), colour in zip(band_masks, BAND_COLOURS):
                        canvas[shrink(band, size)] = colour
                    panels.append((f'rings, {arm}', canvas))
                del band_masks

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
