"""What the ROI and the network each contribute, one point per slide.

Two scatters that ask the question from both sides, IoU at native resolution over
the whole image:

  (a) x = the drawn ROI upsampled and scored alone, y = the held-out model. Does
      the network improve on the ROI it was given?
  (b) x = the same model with its ROI channel replaced by N(0,1) noise, y = the
      model with the ROI provided. Does the ROI add to the network? Both arms
      keep roi_exclusion, so only the tensor the network sees differs.

The diagonal is y = x, so a point above it favours the vertical axis and the
distance from the line is by how much.

IoU only; Dice = 2*IoU/(1+IoU) for the same sets, so it would add no information.

A scatter rather than a paired slope chart: twenty slope lines cross so heavily
that individual slides are unreadable, and the quantity of interest is one number
per slide. The scatter also shows the correlation between ROI quality and model
score, which is why a model handed a poor ROI should not be judged against the
ceiling of a perfect one.

Style follows refrence-figures-hillestad/pig_vs_human_PF_density_Fig1.R: white
panel inside a full #222222 border, sans text in #111111, blue/orange pair, and
filled markers with a black edge so overlapping points stay separable. Gridlines
are kept, which is the one deliberate departure from that script's theme_classic.

    python figures-for-pub/roi-value/figure_roi_value.py \
        --ablation figures-for-pub/roi-value/roi-ablation-noise-2026-10-18_slide_metrics.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'train.py').exists())

# from the reference script's SPECIES_COLORS and theme
SERIES, MISS = '#2B8CBE', '#FF7F0E'
INK, FRAME, GRID, SURFACE = '#111111', '#222222', '#D9D9D9', '#FFFFFF'
BORDER_LW, TICK_LW, MARK_SIZE, MARK_EDGE = 0.75, 0.55, 7.0, 0.85
AXIS_TEXT, AXIS_TITLE, PANEL_TITLE, LEGEND_TEXT = 12.5, 14.4, 16.3, 12.5

# specimen ids carry a capital P for pig; the human cases are AJK*
SPECIES_COLOUR = (('Pig', SERIES), ('Human', MISS))

# diagonal_at places the y = x label where that panel's points leave room; step
# is how far below the data `lo` sits, in units of 0.05 -- a full step where the
# first tick would otherwise land on the spine
PANELS = (dict(letter='a', xlabel='IoU, upsampled ROI vs GT',
               ylabel='IoU, model vs GT',
               diagonal='model = ROI', diagonal_at=0.62, step=0.5),
          dict(letter='b', xlabel='IoU, ROI replaced by noise (retrained)',
               ylabel='IoU with ROI input',
               diagonal='ROI provided = noise', diagonal_at=0.40, step=1.0))


def species(slide):
    return 'Human' if slide.startswith('AJK') else 'Pig'

# fonttype 42 embeds TrueType rather than converting glyphs to paths, which is
# what journals mean by "editable text" in a vector submission
matplotlib.rcParams.update({'font.family': 'sans-serif', 'pdf.fonttype': 42,
                            'ps.fonttype': 42, 'svg.fonttype': 'none'})


def per_slide(path):
    return {s: v for f in json.loads(Path(path).read_text())['folds']
            for s, v in f['per_slide'].items()}


def panel(ax, cfg, x, y, stems):
    lo = np.floor(min(x.min(), y.min()) * 20 - cfg['step']) / 20
    hi = 1.005
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.plot([lo, hi], [lo, hi], color=FRAME, lw=0.9, alpha=0.8, zorder=1)
    mid = lo + cfg['diagonal_at'] * (hi - lo)
    ax.text(mid, mid - 0.025 * (hi - lo), cfg['diagonal'], fontsize=AXIS_TEXT - 1,
            color=FRAME, rotation=45, rotation_mode='anchor', ha='center', va='top')

    # paired, not a two-sample test: both scores come from the same slide. Slides
    # are not fully independent either -- some are halves of one section, several
    # share a heart -- so the p-value is exploratory.
    d = y - x
    q25, q75 = np.percentile(d, [25, 75])
    p = wilcoxon(y, x).pvalue

    up = y > x
    kinds = np.array([species(s) for s in stems])
    for name, colour in SPECIES_COLOUR:
        mask = kinds == name
        ax.plot(x[mask], y[mask], 'o', ms=MARK_SIZE, markerfacecolor=colour,
                markeredgecolor='black', markeredgewidth=MARK_EDGE,
                linestyle='none', zorder=3, label=name)

    # only the exceptions are named: with colour carrying species, an unlabelled
    # point below y = x would be the one thing a reader cannot look up
    for s, xi, yi in zip(np.array(stems)[~up], x[~up], y[~up]):
        ax.annotate(s, (xi, yi), fontsize=AXIS_TEXT - 2.5, color=INK,
                    xytext=(8, -4), textcoords='offset points', zorder=4)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=AXIS_TEXT, colors=INK, color=FRAME,
                   width=TICK_LW, length=4)
    # a closed panel rather than open spines, matching the reference's
    # panel.border; axis.line there is blank for the same reason
    for s in ax.spines.values():
        s.set(visible=True, color=FRAME, linewidth=BORDER_LW)

    ax.set_title(cfg['letter'], fontsize=PANEL_TITLE, color=INK,
                 pad=22, loc='left', fontweight='bold')
    ax.text(0, 1.012,
            f'n = {len(d)}, Wilcoxon p = {p:.1e}, median Δ {np.median(d):+.3f}',
            transform=ax.transAxes, fontsize=AXIS_TEXT, color=FRAME, va='bottom')
    ax.set_xlabel(cfg['xlabel'], fontsize=AXIS_TITLE, color=INK)
    ax.set_ylabel(cfg['ylabel'], fontsize=AXIS_TITLE, color=INK)
    # framed: unboxed, the two legend keys read as two more data points
    ax.legend(loc='lower right', fontsize=LEGEND_TEXT, labelcolor=INK,
              handletextpad=0.4, borderaxespad=0.5, frameon=True,
              facecolor=SURFACE, edgecolor=FRAME, framealpha=1.0).get_frame(
                  ).set_linewidth(BORDER_LW)

    print(f"\n  [{cfg['letter']}]  x {x.mean():.4f}   y {y.mean():.4f}")
    print(f'  gain mean {d.mean():+.4f}  sd {d.std(ddof=1):.4f}  '
          f'range {d.min():+.4f}..{d.max():+.4f}  ({int(up.sum())}/{len(d)} positive)')
    print(f'  median Δ  {np.median(d):+.4f}  IQR [{q25:+.4f}, {q75:+.4f}]  '
          f'Wilcoxon two-sided p = {p:.3e}  (exploratory; slides not independent)')
    print(f'  corr(x, y) r = {np.corrcoef(x, y)[0, 1]:.3f}')
    for s, xi, yi in zip(np.array(stems)[~up], x[~up], y[~up]):
        print(f'  labelled below y = x: {s}  x {xi:.4f}  y {yi:.4f}  '
              f'({species(s)})')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--upsample', default=str(HERE / 'roi-upsample.json'))
    ap.add_argument('--baseline', default=str(ROOT / 'figures-for-pub' / 'model-baseline'
                                              / 'k-fold-bnaug-2026-10-17_slide_metrics.json'))
    ap.add_argument('--ablation', required=True)
    ap.add_argument('--out', default=str(HERE / 'roi-value.pdf'))
    a = ap.parse_args()

    rows = json.loads(Path(a.upsample).read_text())['slides']
    col = lambda k: np.array([r[k] for r in rows], float)

    base, abl = per_slide(a.baseline), per_slide(a.ablation)
    stems = sorted(set(base) & set(abl))
    if set(base) ^ set(abl):
        print(f'  dropped, not in both: {", ".join(sorted(set(base) ^ set(abl)))}')
    fold = lambda d, k: np.array([d[s][k] for s in stems], float)

    series = (([r['stem'] for r in rows], col('roi'), col('model')),
              (stems, fold(abl, 'iou'), fold(base, 'iou')))

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6))
    fig.patch.set_facecolor(SURFACE)
    for ax, cfg, (names, x, y) in zip(axes, PANELS, series):
        panel(ax, cfg, x, y, names)

    # the panels no longer share a y axis, so (b) carries its own label and ticks
    # on the side that used to butt against (a); tight_layout measures them
    fig.tight_layout(w_pad=1.75)
    fig.savefig(a.out, facecolor=SURFACE, bbox_inches='tight')
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
