"""Figure: the model against the ROI it was given, one point per slide.

x is the drawn ROI upsampled and scored alone; y is the held-out model, both at
native resolution over the whole image. The diagonal is y = x, so a point above
it means the network improved on its own ROI and the vertical distance is by how
much.

Written twice, as IoU and as Dice, on IDENTICAL axes so the two panels are read
against the same scale. Dice is what dice_loss optimises and is the convention
in this literature, but it carries no extra information: for the same pair of
sets Dice = 2*IoU/(1+IoU) exactly -- both come from slide_level_metrics.py's own
pixel counts rather than one being derived from the other, but the ranking, the
sign of every delta and the improved-on-ROI count are identical in both panels.
Dice only compresses the scale toward 1.

A scatter rather than a paired slope chart: twenty slope lines over three x
positions cross so heavily that individual slides are unreadable, and the
quantity of interest is one number per slide. The scatter also makes the
correlation between ROI quality and model score visible, which is the reason a
model handed a poor ROI should not be judged against the ceiling of a perfect
one.

Style follows refrence-figures-hillestad/pig_vs_human_PF_density_Fig1.R: white
panel inside a full #222222 border, sans text in #111111, blue/orange pair, and
filled markers with a black edge so overlapping points stay separable. Gridlines
are kept, which is the one deliberate departure from that script's theme_classic.

Usage:
    python figures-for-pub/roi-upsample/figure_roi_upsample.py
"""
# Draft caption:
#
#   Figure N. Segmentation accuracy of the model compared with the region of
#   interest alone. Each point is one slide (n = 20). The horizontal axis gives
#   the intersection over union (A) or Dice coefficient (B) obtained by
#   upsampling the drawn region of interest to the native resolution of the
#   source section and scoring it directly against the manual annotation; the
#   vertical axis gives the corresponding score for the held-out model
#   prediction under leave-one-case-out cross-validation. The diagonal marks
#   equality between the two. Points above the diagonal are slides on which the
#   network improved on the region of interest it was given (19 of 20; mean
#   improvement 0.090 IoU, 0.053 Dice). All scores are computed at the full
#   resolution of the source image and over the entire section.

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

# from the reference script's SPECIES_COLORS and theme
SERIES, MISS = '#2B8CBE', '#FF7F0E'
INK, FRAME, GRID = '#111111', '#222222', '#D9D9D9'
SURFACE = '#FFFFFF'

BORDER_LW, TICK_LW = 0.75, 0.55       # panel.border / axis.ticks linewidths
MARK_SIZE, MARK_EDGE = 7.0, 0.85      # geom_point size 3.2 / stroke 0.85

GAP_PX = 10                            # separation between the two panels
AXIS_TEXT, AXIS_TITLE, PANEL_TITLE, LEGEND_TEXT = 10.0, 11.5, 13.0, 10.0

SHOW_CASE_NAMES = False

# fonttype 42 embeds TrueType rather than converting glyphs to paths, which is
# what journals mean by "editable text" in a vector submission
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', default=str(HERE / 'roi-upsample.json'))
    ap.add_argument('--out', default=str(HERE / 'roi-upsample.pdf'))
    a = ap.parse_args()

    rows = json.loads(Path(a.data).read_text())['slides']
    col = lambda k: np.array([r[k] for r in rows], float)
    panels = (('A. IoU Score', 'IoU', col('roi'), col('model')),
              ('B. Dice Score', 'Dice', col('roi_dice'), col('model_dice')))

    # one pair of limits for both panels, so the two are read against the same
    # scale and Dice's compression toward 1 is visible rather than hidden by a
    # re-zoomed axis
    lo = min(min(r.min(), m.min()) for _, _, r, m in panels)
    lo = np.floor(lo * 20 - 0.5) / 20

    # both panels on one page: they are a single figure with two parts, and a
    # journal wants one file per figure
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (label, name, roi, mod) in zip(axes, panels):
        panel(ax, label, name, roi, mod, rows, lo)

    # one label per axis for the pair, rather than repeating it under each panel
    fig.supxlabel('Upsampled ROI', fontsize=AXIS_TITLE, color=INK)
    fig.supylabel('Model', fontsize=AXIS_TITLE, color=INK)
    # the panels share a scale, so the right one's ticks would only repeat the
    # left one's -- and at a 10 px gap they would land on top of it
    axes[1].tick_params(left=False, labelleft=False)
    # the encoding is identical in both panels, so the legend is shown once
    axes[1].legend(loc='lower right', frameon=False, fontsize=LEGEND_TEXT,
                   labelcolor=INK, handletextpad=0.4, borderaxespad=0.5)

    fig.tight_layout()
    gap_in = GAP_PX / fig.dpi
    fig.subplots_adjust(wspace=gap_in / (axes[0].get_position().width
                                         * fig.get_figwidth()))
    out = Path(a.out)
    fig.savefig(out, facecolor=SURFACE, bbox_inches='tight')
    print(f'wrote {out}')


def panel(ax, label, name, roi, mod, rows, lo):
    ax.set_facecolor(SURFACE)

    hi = 1.005
    ax.grid(color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.plot([lo, hi], [lo, hi], color=FRAME, lw=0.9, alpha=0.8, zorder=1)
    mid = lo + 0.62 * (hi - lo)
    ax.text(mid, mid - 0.025 * (hi - lo), 'model = ROI', fontsize=AXIS_TEXT - 1,
            color=FRAME, rotation=45, rotation_mode='anchor',
            ha='center', va='top')

    up = mod > roi
    for mask, colour, legend in ((up, SERIES, 'model improves on its ROI'),
                                 (~up, MISS, 'model worse than its ROI')):
        ax.plot(roi[mask], mod[mask], 'o', ms=MARK_SIZE, markerfacecolor=colour,
                markeredgecolor='black', markeredgewidth=MARK_EDGE,
                linestyle='none', zorder=3, label=legend)

    if SHOW_CASE_NAMES:
        for r in rows:
            ax.annotate(r['stem'], (r['roi'], r['model']), fontsize=6,
                        color=INK, xytext=(5, -2), textcoords='offset points')

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=AXIS_TEXT, colors=INK, color=FRAME,
                   width=TICK_LW, length=4)
    # a closed panel rather than open spines, matching the reference's
    # panel.border; axis.line there is blank for the same reason
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(FRAME)
        s.set_linewidth(BORDER_LW)

    ax.set_title(label, fontsize=PANEL_TITLE, color=INK, pad=22, loc='left',
                 fontweight='bold')
    ax.text(0, 1.012, f'{name} at native resolution, whole image',
            transform=ax.transAxes, fontsize=AXIS_TEXT, color=FRAME, va='bottom')

    d = mod - roi
    print(f'\n  [{name}]')
    print(f'  roi mean   {roi.mean():.4f}')
    print(f'  model mean {mod.mean():.4f}')
    print(f'  gain mean  {d.mean():+.4f}  sd {d.std(ddof=1):.4f}  '
          f'range {d.min():+.4f}..{d.max():+.4f}  ({int(up.sum())}/{len(d)} positive)')
    print(f'  corr(roi, model) r = {np.corrcoef(roi, mod)[0, 1]:.3f}')


if __name__ == '__main__':
    main()
