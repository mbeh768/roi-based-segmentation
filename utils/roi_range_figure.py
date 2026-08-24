"""Full-range preview of the ROI perturbation used to simulate annotators.

Top strip: the warp at every strength, cropped to the ROI bounding box so the
boundary is legible. Bottom: how much the ROI actually moved, as two separate
panels -- ROI IoU and symmetric surface distance are different measures on
different scales and never share an axis.

Surface distance is the honest axis for this structure. The ROI is thin and
heavily branched, so perimeter-to-area is huge and IoU collapses under
sub-pixel shifts; quoting IoU alone invites comparison against the 0.85-0.95
inter-observer numbers published for compact organs.
"""
import argparse, json, random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage

from data_loader.datasets import _roi_grid_distort

ORIG = '#2a78d6'      # categorical slot 1
PERT = '#eb6834'      # categorical slot 2  (validated pair: CVD dE 24.7)
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d8d7d2'
SCALE = 32            # ROI/global is 256 across an 8192 tiling space


def roi_iou(a, b):
    a, b = a > 0.5, b > 0.5
    return ((a & b).sum().item() + 1e-9) / ((a | b).sum().item() + 1e-9)


def surf_dist(a, b):
    a, b = (a > 0.5).numpy(), (b > 0.5).numpy()
    if not a.any() or not b.any():
        return float('nan')
    ea = a ^ ndimage.binary_erosion(a)
    eb = b ^ ndimage.binary_erosion(b)
    if not ea.any() or not eb.any():
        return float('nan')
    return float((ndimage.distance_transform_edt(~ea)[eb].mean() +
                  ndimage.distance_transform_edt(~eb)[ea].mean()) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--stem', default='AJKV056-7')
    ap.add_argument('--strengths', default='0,0.005,0.01,0.02,0.035,0.05,0.08,0.15,0.3,0.5')
    ap.add_argument('--curve-seeds', type=int, default=12)
    ap.add_argument('--out', default='roi_range_figure.png')
    a = ap.parse_args()

    img_dir = Path(a.data_dir) / 'images'
    g = torch.load(img_dir / a.stem / 'global.pt', weights_only=True)[0].numpy()
    g = (g - g.min()) / (np.ptp(g) + 1e-9)
    roi = torch.load(img_dir / a.stem / f'roi_{a.stem}.pt', weights_only=True)
    S = [float(x) for x in a.strengths.split(',')]

    m = (roi[0] > 0.5).numpy()
    ys, xs = np.where(m)
    pad = 12
    y0, y1 = max(ys.min()-pad, 0), min(ys.max()+pad, m.shape[0])
    x0, x1 = max(xs.min()-pad, 0), min(xs.max()+pad, m.shape[1])

    # curve stats first: the montage captions quote these means, so the strip and
    # the plots below can never disagree (one draw vs a 12-draw mean read as a bug)
    ious, dists = [], []
    for s in S:
        i_, d_ = [], []
        for k in range(a.curve_seeds):
            torch.manual_seed(500+k); random.seed(500+k)
            p_ = roi if s == 0 else _roi_grid_distort(roi.clone(), strength=s)
            i_.append(roi_iou(roi[0], p_[0])); d_.append(surf_dist(roi[0], p_[0]))
        ious.append((np.mean(i_), np.std(i_))); dists.append((np.mean(d_), np.std(d_)))

    fig = plt.figure(figsize=(2.05*len(S), 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1.5], hspace=0.46, wspace=0.20)
    strip = gs[0, :].subgridspec(1, len(S), wspace=0.06)

    for c, s in enumerate(S):
        torch.manual_seed(7); random.seed(7)
        pert = roi if s == 0 else _roi_grid_distort(roi.clone(), strength=s)
        ax = fig.add_subplot(strip[0, c])
        ax.imshow(g[y0:y1, x0:x1], cmap='gray', interpolation='nearest')
        ax.contour(roi[0].numpy()[y0:y1, x0:x1], levels=[0.5], colors=ORIG, linewidths=1.5)
        ax.contour(pert[0].numpy()[y0:y1, x0:x1], levels=[0.5], colors=PERT, linewidths=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(f"{s:g}", fontsize=11, color=INK, pad=6)
        ax.set_xlabel(f"{dists[c][0]*SCALE:.0f} px" if s else "reference",
                      fontsize=8.5, color=INK2, labelpad=3)

    # quantitative panels -- separate axes, never a shared/dual scale
    for col, (data, lab, fmt) in enumerate([
            (ious,  'Agreement with original ROI (IoU)', '{:.2f}'),
            (dists, 'Boundary displacement (slide px)',  '{:.0f}')]):
        ax = fig.add_subplot(gs[1, col])
        mu = np.array([m_ for m_, _ in data]); sd = np.array([s_ for _, s_ in data])
        if col == 1: mu, sd = mu*SCALE, sd*SCALE
        ax.fill_between(S, mu-sd, mu+sd, color=PERT, alpha=0.16, lw=0)
        ax.plot(S, mu, color=PERT, lw=2, marker='o', ms=5,
                markerfacecolor=PERT, markeredgecolor='white', markeredgewidth=1.2)
        ax.set_title(lab, fontsize=10, color=INK, pad=8, loc='left')
        ax.set_xlabel('distortion strength', fontsize=9, color=INK2)
        ax.grid(True, color=GRID, lw=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        for sp in ('top','right'): ax.spines[sp].set_visible(False)
        for sp in ('left','bottom'): ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8.5)
        for x, y in [(S[i], mu[i]) for i in (1, 4, 6)]:
            ax.annotate(fmt.format(y), (x, y), textcoords='offset points',
                        xytext=(0, 9), ha='center', fontsize=8, color=INK2)

    fig.suptitle(f"ROI perturbation across the full strength range — {a.stem}",
                 fontsize=13, color=INK, y=0.985)
    fig.text(0.5, 0.935, "blue = original annotation      orange = simulated second annotator"
             "      (label under each panel = mean boundary displacement at slide scale)",
             ha='center', fontsize=9.5, color=INK2)
    fig.savefig(a.out, dpi=140, bbox_inches='tight', facecolor='#fcfcfb')
    print(f"wrote {a.out}\n")
    print("%9s %10s %8s %14s" % ("strength","ROI IoU","sd","slide-scale px"))
    for s,(i_,isd),(d_,dsd) in zip(S, ious, dists):
        print("%9.3f %10.4f %8.4f %10.0f +/-%.0f" % (s, i_, isd, d_*SCALE, dsd*SCALE))
    json.dump({'strengths':S,'iou':[list(map(float,x)) for x in ious],
               'surf_px':[[float(x[0]*SCALE), float(x[1]*SCALE)] for x in dists]},
              open(Path(a.out).with_suffix('.json'),'w'), indent=2)


if __name__ == '__main__':
    main()
