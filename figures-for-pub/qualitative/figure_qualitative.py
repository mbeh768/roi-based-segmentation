"""Qualitative results panel: the section, beside what the model got wrong.

Each slide is one image -- the raw section, then the agreement map, separated by
a 2 px gap at identical scale and extent.

The ROI is deliberately NOT shown, though export_canvases.py stores it. It is
256x256 against an 8192-wide canvas: 32x coarser, 1024x fewer pixels. At figure
scale that upsamples into a smooth silhouette resembling the answer, so the
panel would suggest the ROI hands the model its result while hiding the
resolution gap that is the actual point. The dependence on the ROI belongs in a
measurement -- the perturbation sweep -- not in a picture that overstates it.

    white   agreement       pred AND gt
    red     disagreement    pred XOR gt, either direction
    black   background      covered by a tile, neither pred nor gt
    grey    not scored      no tile reached it, excluded from the IoU

Red is the symmetric difference, not pred\\gt. Painting only false positives
would leave the misses white and under-segmentation would vanish: 17P645-1
carries 181k missed pixels against 47k false ones, so the weakest slide would
show less red than the best one. Grey is separate from black because coverage
runs 5-38% and painting unscored slide as correct background claims credit for
area the model never saw.

The left panel is the raw stained section as the canvases export rendered it.

Slides run best to worst, each from a distinct case.

Usage:
    python figures-for-pub/qualitative/figure_qualitative.py
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent

SLIDES = ['21PControl-9-2of2', '17P391_7A', 'AJKY121-7-2', '17P645-1']

# indexed by the code array below: 0 unscored, 1 background, 2 agree, 3 disagree
LUT = np.array([(52, 52, 49), (0, 0, 0), (255, 255, 255), (230, 103, 103)], np.uint8)
LABELS = ['Not scored', 'Background', 'Agreement', 'Disagreement']
SURFACE, INK, INK_2 = '#fcfcfb', '#0b0b0b', '#52514e'
SURFACE_RGB = (252, 252, 251)   # the 2 px gap between the pair, as pixels
WIDTH = 7.4                      # inches, single-column-spanning figure

# Case identifiers are internal nomenclature -- '21PControl-9-2of2' tells a
# reader nothing. The IoU stays, because it is what orders the rows. Set True to
# restore the names when the figure needs matching against a supplementary table.
SHOW_CASE_NAMES = False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--canvases', default=str(HERE / 'canvases'))
    ap.add_argument('--out', default=str(HERE / 'qualitative-results.png'))
    ap.add_argument('--dpi', type=int, default=1200)
    ap.add_argument('--slides', nargs='*', default=SLIDES)
    a = ap.parse_args()

    # the pair is one image, not two axes: that is what keeps the gap at exactly
    # 2 px instead of whatever subplot spacing works out to
    pairs, used = [], set()
    for stem in a.slides:
        d = np.load(Path(a.canvases) / f'{stem}.npz')
        pred, gt = d['pred'], d['gt']
        code = d['covered'].astype(np.uint8)
        code[pred & gt] = 2
        code[pred ^ gt] = 3
        used.update(np.unique(code).tolist())
        gap = np.full((code.shape[0], 2, 3), SURFACE_RGB, np.uint8)
        pairs.append((stem, float(d['iou']), float(d['dice']),
                      np.hstack([d['image'], gap, LUT[code]])))

    # the figure is sized from the pairs rather than the other way round. imshow
    # holds aspect, so a cell taller than its image gets margin above and below
    # -- the vertical gap between rows is that margin, not subplot spacing.
    rows, cols = len(pairs) // 2, 2
    aspect = max(p.shape[0] / p.shape[1] for *_, p in pairs)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(WIDTH, rows * (WIDTH / cols) * aspect),
                             gridspec_kw={'hspace': 0.5, 'wspace': 0.03})
    fig.patch.set_facecolor(SURFACE)

    for ax, (stem, iou, dice, pair) in zip(axes.ravel(), pairs):
        ax.imshow(pair)
        score = f'IoU {iou:.3f}   Dice {dice:.3f}'
        ax.set_title(f'{stem}   {score}' if SHOW_CASE_NAMES else score,
                     fontsize=8, color=INK)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # a swatch for a colour that never appears just invites the reader to hunt
    # for it -- the unguided arm scores the whole canvas, so it has no grey
    keys = [i for i in range(len(LABELS)) if i in used]
    fig.legend([Rectangle((0, 0), 1, 1, facecolor=LUT[i] / 255, edgecolor='#c3c2b7',
                          lw=0.4) for i in keys],
               [LABELS[i] for i in keys], loc='lower center', ncol=len(keys),
               frameon=False, fontsize=8, labelcolor=INK_2)
    fig.savefig(a.out, dpi=a.dpi, facecolor=SURFACE, bbox_inches='tight')
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
