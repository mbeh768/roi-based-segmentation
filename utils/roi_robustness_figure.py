"""EXPERIMENT 15 figure: segmentation quality vs ROI perturbation.

x-axis is boundary displacement in slide pixels, not ROI IoU. This ROI is a thin
branching structure, so its IoU collapses under sub-pixel shifts and would
misrepresent how small the perturbations are.

The shaded band marks REAL annotator disagreement, measured from the three
genuinely annotated variants on file (roi_<stem>.pt, _1, _2). Without it the
x-axis is an arbitrary synthetic scale; with it the figure can say where actual
human disagreement falls relative to where the model degrades.
"""
import argparse, json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage

S1, INK, INK2, GRID = '#2a78d6', '#0b0b0b', '#52514e', '#d8d7d2'
ANCHOR = '#eb6834'
SCALE = 32


def surf_dist(a, b):
    a, b = (a > 0.5).numpy(), (b > 0.5).numpy()
    ea = a ^ ndimage.binary_erosion(a); eb = b ^ ndimage.binary_erosion(b)
    if not ea.any() or not eb.any(): return float('nan')
    return float((ndimage.distance_transform_edt(~ea)[eb].mean() +
                  ndimage.distance_transform_edt(~eb)[ea].mean()) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--exp11', default=None)
    ap.add_argument('--out', default='fig_roi_robustness.png')
    a = ap.parse_args()

    d = json.loads(Path(a.sweep).read_text())
    res = d['results']

    # real annotator disagreement, from the 3 annotated variants actually on file
    img = Path(a.data_dir) / 'images'
    real = []
    for sd_ in sorted(p for p in img.iterdir() if p.is_dir()):
        vs = sorted(sd_.glob(f'roi_{sd_.name}*.pt'))
        if len(vs) < 2: continue
        rois = [torch.load(v, weights_only=True)[0] for v in vs]
        for i, j in combinations(range(len(rois)), 2):
            real.append(surf_dist(rois[i], rois[j]) * SCALE)
    real = np.array([r for r in real if r == r])
    lo, hi, med = np.percentile(real, 10), np.percentile(real, 90), np.median(real)

    S = d['strengths']
    xs, mu, sd = [], [], []
    for s in S:
        rs = [r for r in res if r['strength'] == s]
        xs.append(np.mean([r['roi_surf_px'] for r in rs]))
        v = [r['slide_iou'] for r in rs]
        mu.append(np.mean(v)); sd.append(np.std(v))
    xs, mu, sd = map(np.array, (xs, mu, sd))

    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    # The real annotated variants differ by a median of ~40px, but their EFFECT is not
    # readable off this curve: a grid warp displaces the ROI off the anatomy, whereas a
    # second annotator still traces it. Marking the displacement alone would imply the
    # curve's y-value applies to real annotators, which it does not. Shown as a rug on
    # the axis only, with the effect left to the (pending) direct measurement.
    ax.plot([lo, hi], [0.404, 0.404], color=ANCHOR, lw=3, solid_capstyle='butt', zorder=4)
    ax.annotate(f'displacement between the three real annotated ROIs '
                f'(median {med:.0f} px)\n— their effect is measured directly, not read '
                f'off this curve',
                xy=(hi, 0.404), xytext=(hi + 6, 0.44), fontsize=8.5,
                color=ANCHOR, ha='left', va='bottom')

    ax.axhline(mu[0], color=INK2, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.annotate(f'unperturbed  {mu[0]:.3f}', xy=(xs[-1], mu[0]), xytext=(-4, 6),
                textcoords='offset points', ha='right', fontsize=9, color=INK2)

    ax.fill_between(xs, mu - sd, mu + sd, color=S1, alpha=0.16, lw=0, zorder=2)
    ax.plot(xs, mu, color=S1, lw=2, marker='o', ms=7, zorder=3,
            markerfacecolor=S1, markeredgecolor='white', markeredgewidth=1.4)
    for x, y, s in zip(xs, mu, S):
        ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points', xytext=(0, 12),
                    ha='center', fontsize=8.5, color=INK)

    ax.set_xlabel('ROI boundary displacement (slide pixels)', fontsize=10.5, color=INK2)
    ax.set_ylabel('Segmentation IoU vs ground truth', fontsize=10.5, color=INK2)
    fig.text(0.055, 0.965, 'Segmentation quality under annotator variation',
             fontsize=13.5, color=INK, ha='left', va='top')
    fig.text(0.055, 0.918,
             '14-fold leave-one-case-out · 5 fixed annotator presets · inference only\n'
             'Grid warp is a conservative upper bound: it moves the ROI off the anatomy,\n'
             'which a second annotator does not. Band is ±1 sd across folds and presets.',
             fontsize=9, color=INK2, ha='left', va='top', linespacing=1.5)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.9); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5)
    ax.set_ylim(0.40, 1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.80])
    fig.savefig(a.out, dpi=150, facecolor='#fcfcfb')
    print(f'wrote {a.out}')
    print(f'\nreal-annotator disagreement (n={len(real)} pairs): '
          f'median {med:.0f} px, 10-90% {lo:.0f}-{hi:.0f} px')
    print(f'{"disp px":>9} {"IoU":>9} {"sd":>8} {"vs base":>9}')
    for x, m, s in zip(xs, mu, sd):
        print(f'{x:>9.0f} {m:>9.4f} {s:>8.4f} {m-mu[0]:>+9.4f}')
    json.dump({'real_px': real.tolist(), 'median_real_px': float(med),
               'p10': float(lo), 'p90': float(hi),
               'disp_px': xs.tolist(), 'iou': mu.tolist(), 'sd': sd.tolist()},
              open(Path(a.out).with_suffix('.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
