"""Figure A -- the ROI as an input prior.

Paired slope chart: slide-level IoU for each held-out case, baseline (real ROI)
against the ablation that replaces the ROI with N(0,1) noise. Both arms are
scored on the IDENTICAL tile set and canvas -- the ROI still selects tiles in
the ablation, only the tensor the model sees changes -- so the two points for a
case are directly comparable and a connecting line is meaningful.

That is why the unguided arm is NOT on this figure. Removing tile selection
changes the evaluation population as well as the training (961 tiles over the
whole canvas, rather than ~300 over the ROI-adjacent region), so a third point
would imply a comparability the numbers do not have. It gets its own figure.

The figure plots data only -- no mean line, no summary statistics. Every case
moves the same direction, so a mean would restate what fourteen lines already
show, and a summary drawn as a line reads as a fifteenth case. Aggregates go in
the prose; the script prints them to stdout so the two cannot disagree.

Usage:
    python figures-for-pub/roi-ablation/figure_a_slope.py
    python figures-for-pub/roi-ablation/figure_a_slope.py --out /tmp/fig_a.png
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

# Style follows refrence-figures-hillestad/pig_vs_human_PF_density_Fig1.R: white
# panel inside a full #222222 border, sans text in #111111, and filled markers
# with a black edge. Gridlines are kept, the one departure from theme_classic.
SERIES = '#2B8CBE'
INK, FRAME, GRID = '#111111', '#222222', '#D9D9D9'
SURFACE = '#FFFFFF'

BORDER_LW, TICK_LW = 0.75, 0.55
MARK_SIZE, MARK_EDGE = 5.0, 0.8
AXIS_TEXT, AXIS_TITLE, PANEL_TITLE = 10.0, 11.5, 13.0

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})

# Case identifiers are internal nomenclature -- 'AJKY121-7-2' tells a reader
# nothing, and fourteen of them is a lot of ink for no information. The deltas
# stay, because reading an exact drop off a slope is hard. Set True to restore
# the names when the figure needs to be matched against a supplementary table.
SHOW_CASE_NAMES = False


def load(path):
    folds = json.loads(Path(path).read_text())['folds']
    return {f['case'][0]: f['slide_iou_case'] for f in folds}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--baseline', default=str(ROOT / 'figures-for-pub' / 'baseline'
                                              / 'k-fold-baseline-2026-09-02_slide_metrics.json'))
    ap.add_argument('--ablation', default=str(
        HERE / 'roi-ablation-noise-2026-09-03_slide_metrics.json'))
    ap.add_argument('--out', default=str(HERE / 'noise-ablate.png'))
    ap.add_argument('--dpi', type=int, default=300)
    a = ap.parse_args()

    base, abl = load(a.baseline), load(a.ablation)
    missing = set(base) ^ set(abl)
    if missing:
        raise SystemExit(f'cases present in only one arm: {sorted(missing)}')
    cases = sorted(base, key=lambda c: abl[c] - base[c])   # steepest drop first

    # wider than tall-and-narrow: a cramped x-span exaggerates the slopes, and
    # the reader is meant to judge the drop from the axis, not from the angle
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x0, x1 = 0.0, 1.0

    # lines carry the alpha so the crossings stay readable; the markers are
    # drawn separately at full opacity so no endpoint is dimmed by overlap
    for c in cases:
        ax.plot([x0, x1], [base[c], abl[c]], color=SERIES, lw=1.5, alpha=0.75,
                zorder=2, solid_capstyle='round')
    for c in cases:
        ax.plot([x0, x1], [base[c], abl[c]], linestyle='none', marker='o',
                markersize=MARK_SIZE, markerfacecolor=SERIES,
                markeredgecolor='black', markeredgewidth=MARK_EDGE,
                alpha=1.0, zorder=3)

    # every case labelled at its ablation endpoint. Endpoints collide (four
    # cases land within 0.03 of each other), so nudge overlapping labels apart
    # vertically and draw a leader to the true position rather than moving the
    # point or letting the text stack.
    order = sorted(cases, key=lambda c: abl[c])
    ys, min_gap = [], 0.023
    for c in order:
        y = abl[c]
        if ys and y - ys[-1] < min_gap:
            y = ys[-1] + min_gap
        ys.append(y)
    # Leaders are drawn by hand rather than with annotate's arrowprops, which
    # anchors to whichever point on the text bbox is nearest the marker -- so a
    # nudged label gets its line at a corner and an un-nudged one gets it at the
    # middle, and the anchor point wanders down the column. These key to the
    # left edge of the text box every time.
    label_x, gap = x1 + 0.065, 0.010
    for c, y in zip(order, ys):
        ax.plot([x1, label_x - gap], [abl[c], y], color=FRAME, lw=0.6,
                alpha=0.55, zorder=1, solid_capstyle='butt')
        label = f'{c}   {abl[c] - base[c]:+.3f}' if SHOW_CASE_NAMES \
            else f'{abl[c] - base[c]:+.3f}'
        ax.text(label_x, y, label,
                ha='left', va='center', fontsize=8.0, color=INK)

    ax.set_xlim(-0.18, 1.34)
    ax.set_ylim(0.30, 1.02)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(['Baseline', 'Noise'], fontsize=AXIS_TITLE, color=INK)
    ax.set_ylabel('Slide-level IoU', fontsize=AXIS_TITLE, color=INK)
    ax.tick_params(axis='y', labelsize=AXIS_TEXT, colors=INK, color=FRAME,
                   width=TICK_LW, length=4)
    ax.tick_params(axis='x', length=0)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(FRAME)
        sp.set_linewidth(BORDER_LW)
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_axisbelow(True)

    ax.set_title('Replacing the ROI with noise, tile selection unchanged',
                 fontsize=PANEL_TITLE, color=INK, pad=26, loc='left')
    ax.text(0, 1.012, f'{len(cases)} leave-one-case-out folds',
            transform=ax.transAxes, fontsize=AXIS_TEXT, color=FRAME, va='bottom')

    fig.tight_layout()
    out = Path(a.out).with_suffix('.pdf')
    fig.savefig(out, facecolor=SURFACE, bbox_inches='tight')
    print(f'wrote {out}')

    # stdout only -- the numbers for the prose, kept out of the figure
    d = np.array([abl[c] - base[c] for c in cases])
    print(f"\n  {'case':<16}{'baseline':>10}{'ablation':>10}{'delta':>9}")
    for c in cases:
        print(f'  {c:<16}{base[c]:>10.4f}{abl[c]:>10.4f}{abl[c] - base[c]:>+9.4f}')
    print(f'\n  baseline mean {np.mean([base[c] for c in cases]):.4f}   '
          f'ablation mean {np.mean([abl[c] for c in cases]):.4f}')
    print(f'  delta mean {d.mean():+.4f}  sd {d.std(ddof=1):.4f}  '
          f'range {d.min():+.4f}..{d.max():+.4f}  ({int((d < 0).sum())}/{len(d)} negative)')


if __name__ == '__main__':
    main()
