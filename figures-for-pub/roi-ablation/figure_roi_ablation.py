"""x = model with its ROI channel replaced by N(0,1) noise, y = same model with
the drawn ROI, one point per slide. Mirror of roi-upsample/figure_roi_upsample.py:
that asks if the network adds to the ROI, this asks if the ROI adds to the
network. Both arms keep roi_exclusion, so only the tensor the network sees differs.

    python figures-for-pub/roi-ablation/figure_roi_ablation.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'train.py').exists())

SERIES, MISS = '#2B8CBE', '#FF7F0E'
INK, FRAME, GRID, SURFACE = '#111111', '#222222', '#D9D9D9', '#FFFFFF'
BORDER_LW, TICK_LW, MARK_SIZE, MARK_EDGE = 0.75, 0.55, 7.0, 0.85
AXIS_TEXT, AXIS_TITLE, PANEL_TITLE, LEGEND_TEXT = 10.0, 11.5, 13.0, 10.0

matplotlib.rcParams.update({'font.family': 'sans-serif', 'pdf.fonttype': 42,
                            'ps.fonttype': 42, 'svg.fonttype': 'none'})


def per_slide(path):
    return {s: v for f in json.loads(Path(path).read_text())['folds']
            for s, v in f['per_slide'].items()}


def panel(ax, label, name, noise, real, lo):
    hi = 1.005
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.plot([lo, hi], [lo, hi], color=FRAME, lw=0.9, alpha=0.8, zorder=1)
    mid = lo + 0.62 * (hi - lo)
    ax.text(mid, mid - 0.025 * (hi - lo), 'drawn ROI = noise', fontsize=AXIS_TEXT - 1,
            color=FRAME, rotation=45, rotation_mode='anchor', ha='center', va='top')

    up = real > noise
    for mask, colour, legend in ((up, SERIES, 'better with ROI input'),
                                 (~up, MISS, 'better without ROI input')):
        ax.plot(noise[mask], real[mask], 'o', ms=MARK_SIZE, markerfacecolor=colour,
                markeredgecolor='black', markeredgewidth=MARK_EDGE,
                linestyle='none', zorder=3, label=legend)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=AXIS_TEXT, colors=INK, color=FRAME, width=TICK_LW, length=4)
    for s in ax.spines.values():
        s.set(visible=True, color=FRAME, linewidth=BORDER_LW)
    ax.set_title(label, fontsize=PANEL_TITLE, color=INK, pad=22, loc='left',
                 fontweight='bold')
    ax.text(0, 1.012, f'{name} at native resolution, whole image',
            transform=ax.transAxes, fontsize=AXIS_TEXT, color=FRAME, va='bottom')

    d = real - noise
    print(f'\n  [{name}]  noise {noise.mean():.4f}   ROI {real.mean():.4f}')
    print(f'  gain mean {d.mean():+.4f}  sd {d.std(ddof=1):.4f}  '
          f'range {d.min():+.4f}..{d.max():+.4f}  ({int(up.sum())}/{len(d)} positive)')
    print(f'  corr(noise, ROI) r = {np.corrcoef(noise, real)[0, 1]:.3f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--baseline', default=str(ROOT / 'figures-for-pub' / 'model-baseline'
                                              / 'k-fold-bnaug-2026-10-17_slide_metrics.json'))
    ap.add_argument('--ablation', required=True)
    ap.add_argument('--out', default=str(HERE / 'roi-ablation.pdf'))
    a = ap.parse_args()

    base, abl = per_slide(a.baseline), per_slide(a.ablation)
    stems = sorted(set(base) & set(abl))
    if set(base) ^ set(abl):
        print(f'  dropped, not in both: {", ".join(sorted(set(base) ^ set(abl)))}')

    col = lambda d, k: np.array([d[s][k] for s in stems], float)
    panels = (('A. IoU Score', 'IoU', col(abl, 'iou'), col(base, 'iou')),
              ('B. Dice Score', 'Dice', col(abl, 'dice'), col(base, 'dice')))
    # a full 0.05 step below the data, not half: when `lo` lands on a multiple of
    # the 0.1 tick spacing the first tick sits on the spine and its label runs
    # into the neighbouring panel's last one
    lo = np.floor(min(min(x.min(), y.min()) for _, _, x, y in panels) * 20 - 1.0) / 20

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, args in zip(axes, panels):
        panel(ax, *args, lo)
    fig.supxlabel('ROI replaced by noise', fontsize=AXIS_TITLE, color=INK)
    fig.supylabel('Drawn ROI', fontsize=AXIS_TITLE, color=INK)
    axes[1].tick_params(left=False, labelleft=False)
    axes[1].legend(loc='lower right', frameon=False, fontsize=LEGEND_TEXT,
                   labelcolor=INK, handletextpad=0.4, borderaxespad=0.5)

    fig.tight_layout()
    fig.subplots_adjust(wspace=(10 / fig.dpi) / (axes[0].get_position().width
                                                 * fig.get_figwidth()))
    fig.savefig(a.out, facecolor=SURFACE, bbox_inches='tight')
    print(f'\nwrote {a.out}  (n = {len(stems)})')


if __name__ == '__main__':
    main()
