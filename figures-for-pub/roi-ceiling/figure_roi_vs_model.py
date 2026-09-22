"""Figure: the model against the ROI it was given, one point per slide.

x is the drawn ROI upsampled and scored alone; y is the held-out model, both at
native resolution over the whole image. The diagonal is y = x, so a point above
it means the network improved on its own ROI and the vertical distance is by how
much.

Written twice, as IoU and as Dice. Dice is what dice_loss optimises and is the
convention in this literature, but it carries no extra information: for the same
pair of sets Dice = 2*IoU/(1+IoU) exactly, so it is derived rather than
recomputed, and the ranking, the sign of every delta and the improved-on-ROI
count are identical in both panels. Dice only compresses the scale toward 1.

A scatter rather than a paired slope chart: twenty slope lines over three x
positions cross so heavily that individual slides are unreadable, and the
quantity of interest is one number per slide. The scatter also makes the
correlation between ROI quality and model score visible, which is the reason a
model handed a poor ROI should not be judged against the ceiling of a perfect
one.

Usage:
    python figures-for-pub/roi-ceiling/figure_roi_vs_model.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

SERIES, MISS = '#2a78d6', '#eb6834'
INK, INK_2, INK_MUTED = '#0b0b0b', '#52514e', '#8a8985'
SURFACE = '#fcfcfb'

SHOW_CASE_NAMES = False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', default=str(HERE / 'roi_ceiling.json'))
    ap.add_argument('--outdir', default=str(HERE))
    ap.add_argument('--dpi', type=int, default=300)
    a = ap.parse_args()

    rows = json.loads(Path(a.data).read_text())['slides']
    iou_roi = np.array([r['roi'] for r in rows])
    iou_mod = np.array([r['model'] for r in rows])
    dice = lambda x: 2 * x / (1 + x)

    for name, roi, mod, fname in (
            ('IoU', iou_roi, iou_mod, 'roi-vs-model.png'),
            ('Dice', dice(iou_roi), dice(iou_mod), 'roi-vs-model-dice.png')):
        panel(name, roi, mod, rows, Path(a.outdir) / fname, a.dpi)


def panel(name, roi, mod, rows, out, dpi):

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # limits from the data, since Dice compresses everything toward 1 and a
    # shared axis would leave its points crowded in a corner
    lo = np.floor(min(roi.min(), mod.min()) * 20 - 0.5) / 20
    hi = 1.005
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, lw=1.0, alpha=0.7, zorder=1)
    mid = lo + 0.62 * (hi - lo)
    ax.text(mid, mid - 0.025 * (hi - lo), f'model = ROI', fontsize=8,
            color=INK_MUTED, rotation=45, rotation_mode='anchor',
            ha='center', va='top')

    up = mod > roi
    ax.plot(roi[up], mod[up], 'o', ms=7, color=SERIES, markeredgecolor=SURFACE,
            markeredgewidth=0.9, zorder=3, label='model improves on its ROI')
    ax.plot(roi[~up], mod[~up], 'o', ms=7, color=MISS, markeredgecolor=SURFACE,
            markeredgewidth=0.9, zorder=3, label='model worse than its ROI')

    if SHOW_CASE_NAMES:
        for r in rows:
            ax.annotate(r['stem'], (r['roi'], r['model']), fontsize=6,
                        color=INK_2, xytext=(5, -2), textcoords='offset points')

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel(f'Drawn ROI alone, no network  ({name})', fontsize=10, color=INK)
    ax.set_ylabel(f'Model  ({name})', fontsize=10, color=INK)
    ax.tick_params(labelsize=9, colors=INK_2, length=3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK_MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(color=INK_MUTED, alpha=0.2, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False, fontsize=8.5, labelcolor=INK_2,
              handletextpad=0.4, borderaxespad=0.4)

    d = mod - roi
    ax.set_title(f'The model improves on its ROI on {int(up.sum())} of {len(rows)} '
                 f'slides', fontsize=11, color=INK, pad=24, loc='left')
    ax.text(0, 1.012, f'{name} at native resolution, whole image',
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va='bottom')

    fig.tight_layout()
    fig.savefig(out, dpi=dpi, facecolor=SURFACE, bbox_inches='tight')
    print(f'wrote {out}')

    print(f'\n  [{name}]')
    print(f'  roi mean   {roi.mean():.4f}')
    print(f'  model mean {mod.mean():.4f}')
    print(f'  gain mean  {d.mean():+.4f}  sd {d.std(ddof=1):.4f}  '
          f'range {d.min():+.4f}..{d.max():+.4f}  ({int(up.sum())}/{len(d)} positive)')
    print(f'  corr(roi, model) r = {np.corrcoef(roi, mod)[0, 1]:.3f}')


if __name__ == '__main__':
    main()
