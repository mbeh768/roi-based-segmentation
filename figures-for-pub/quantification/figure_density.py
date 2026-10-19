"""Figure: relative MYL4 density by depth, manual annotation against prediction.

Geometry and theme follow refrence-figures-hillestad/pig_vs_human_PF_density_Fig1.R,
with the paired factor changed from pig/human to ground truth/predicted.

Agreement per band is Bland-Altman: bias, 95% limits of agreement, and mean
absolute difference as a share of the biological effect. Not Lin's concordance,
which divides by the between-slide spread and so reports a signal-to-noise
ratio rather than closeness -- it reads a band as poor agreement when the effect
is small, even where the absolute error did not move. No significance test: the
claim is that the two arms measure the same thing, which a difference test can
only fail to reject.

The effect a band's error is measured against is the near band minus the deep
band ON THE ANNOTATION, per species, so it is a property of the biology and not
of the arm being judged.

rho is normalised per slide by `scale`, the channel ratio over the tissue
outside the mask. Under the ratio of sums that put the union of bands at exactly
0.5; a mean of per-pixel ratios is not pinned there, so no reference line is
drawn. Split by species because that spread would otherwise swamp the comparison.

One section is excluded, AJKY121-8-2, and it is named rather than greyed: a
section is either measurable and plotted, or excluded and listed. Every other
section is in every statistic. See EXCLUSIONS.md beside this file for the reason
and the measurement behind it.

Usage:
    python figures-for-pub/quantification/figure_density.py
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'train.py').exists())

# from the reference script's SPECIES_COLORS / SPECIES_SHAPES and theme
ARMS = (('ground truth', 'Ground truth', '#2B8CBE', 'o'),
        ('predicted', 'Predicted', '#FF7F0E', 's'))
BANDS = ['0-99 um', '100-199 um', '>200 um']
LABELS = ['0–99 µm', '100–199 µm', '>200 µm']

INK, INK_2 = '#111111', '#52514e'
FLAG_GREY = '#c3c2b7'     # flagged sections, so shape carries the meaning not colour
FRAME, GRID, SURFACE = '#222222', '#D9D9D9', '#FFFFFF'
BOX_EDGE, BOX_ALPHA = '#7A7A7A', 0.22
BORDER_LW, TICK_LW, BOX_LW = 0.75, 0.55, 0.60
MARK_SIZE, MARK_EDGE = 6.5, 0.85
AXIS_TEXT, AXIS_TITLE, LEGEND_TEXT, STAR_TEXT = 10.0, 11.5, 10.0, 12.0
PANEL_TITLE = 13.0

# specimen ids carry a capital P for pig; the human cases are AJK*
SPECIES = (('A. Pig', lambda s: 'P' in s), ('B. Human', lambda s: 'P' not in s))

# Dropped from the figure and from every statistic. The measurement is a mean of
# per-pixel ratios, and on this section 98% of tissue pixels sit at exactly zero
# in MYL4 -- see EXCLUSIONS.md beside this file.
EXCLUDED = ('AJKY121-8-2',)

# Previously held sections drawn in grey and kept in the statistics. Emptied:
# greying a point asks the reader to discount it while still counting it. A
# section now either measures or is excluded by name.
MARKED = {}

DODGE, BOX_WIDTH = 0.52, 0.30
OVERLAP = 0.40        # adjacent fanned points overlap by this much of a marker

# fonttype 42 embeds TrueType rather than converting glyphs to paths
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})


LOA_Z = 1.96          # 95% limits of agreement


def bland_altman(gt, pred):
    """Bias and 95% limits of agreement for a paired difference.

    Absolute, in the units of the measurement, so two bands with different
    spreads stay comparable -- which is the property CCC gives up.
    """
    d = gt - pred
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    return bias, sd, bias - LOA_Z * sd, bias + LOA_Z * sd, float(np.abs(d).mean())


def offsets(values, span, tol):
    """Fan colliding points sideways, alternating about the centre. Points
    within `tol` in y would overplot, so each takes the next free slot. No
    jitter, so the figure is identical on every run."""
    out = np.zeros(len(values))
    order = np.argsort(values, kind='stable')
    group = []
    for idx in order:
        if group and values[idx] - values[group[0]] > tol:
            group = []
        # slots run 0, +1, -1, +2, -2, ... so a cluster stays centred
        k = len(group)
        out[idx] = span * ((k + 1) // 2) * (1 if k % 2 else -1) if k else 0.0
        group.append(idx)
    return out


def load(path):
    """{band: {arm: {slide: rho}}}, from quantify_density.py's CSV."""
    out = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(path)):
        out[r['band']][r['arm']][r['slide']] = float(r['rho_px'])
    return out


def panel(ax, ax_title, data, slides, lo, hi):
    """One species: dodged boxes per band, points fanned, bracket and stars."""
    rows, groups = [], []
    for i, band in enumerate(BANDS):
        paired = {arm: np.array([data[band][arm][s] for s in slides])
                  for arm, *_ in ARMS}
        for j, (arm, _, colour, marker) in enumerate(ARMS):
            x = i + (j - 0.5) * DODGE
            # whiskers to the extremes: every point is drawn, so a 1.5 IQR
            # whisker would leave points stranded outside it
            bp = ax.boxplot([paired[arm]], positions=[x], widths=BOX_WIDTH,
                            whis=(0, 100), showfliers=False, patch_artist=True,
                            manage_ticks=False)
            for part in ('whiskers', 'caps', 'medians', 'boxes'):
                for artist in bp[part]:
                    artist.set(color=BOX_EDGE, linewidth=BOX_LW)
            bp['boxes'][0].set(facecolor=colour, alpha=BOX_ALPHA, edgecolor=BOX_EDGE)
            groups.append((x, paired[arm], colour, marker, slides))

        bias, sd, loa_lo, loa_hi, mad = bland_altman(paired['ground truth'],
                                                     paired['predicted'])
        # every section, marked ones included: a marker identifies them, it does
        # not remove them from the statistic
        rows.append({'species': ax_title.split('. ')[1], 'band': band,
                     'n': len(slides),
                     'gt_mean': paired['ground truth'].mean(),
                     'pred_mean': paired['predicted'].mean(),
                     'bias': bias, 'sd_difference': sd,
                     'loa_lower': loa_lo, 'loa_upper': loa_hi,
                     'loa_width': loa_hi - loa_lo,
                     'mean_abs_difference': mad})

    # near band minus deep band ON THE ANNOTATION: the effect the error is
    # judged against has to be a property of the biology, not of the arm
    effect = rows[0]['gt_mean'] - rows[-1]['gt_mean']
    for r in rows:
        r['effect'] = effect
        r['pct_of_effect'] = 100 * r['mean_abs_difference'] / effect if effect else float('nan')

    pad = 0.10 * (hi - lo)
    y = hi + 0.45 * pad
    for i, r in enumerate(rows):
        ax.text(i, y, f"bias {r['bias']:+.3f}\n95% LoA ±{LOA_Z * r['sd_difference']:.3f}",
                ha='center', va='bottom', fontsize=STAR_TEXT - 2.0, color=INK_2,
                linespacing=1.4)

    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels(LABELS, fontsize=AXIS_TITLE, color=INK)
    ax.tick_params(axis='x', length=0, pad=8)
    ax.tick_params(axis='y', labelsize=AXIS_TEXT, colors=INK, color=FRAME,
                   width=TICK_LW, length=5)
    ax.set_xlim(-0.65, len(BANDS) - 0.35)
    ax.set_ylim(lo - pad, hi + 1.9 * pad)
    for sp in ax.spines.values():
        sp.set(visible=True, color=FRAME, linewidth=BORDER_LW)
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title(f'{ax_title}  (n = {len(slides)})', fontsize=PANEL_TITLE, color=INK,
                 pad=12, loc='left', fontweight='bold')
    return rows, groups


def draw_points(ax, groups):
    """Fanned last: spacing points by marker size needs the final axes size."""
    box = ax.get_window_extent()
    per_pt = ax.figure.dpi / 72.0
    dx = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (box.width / per_pt)
    dy = (ax.get_ylim()[1] - ax.get_ylim()[0]) / (box.height / per_pt)
    for x, v, colour, marker, slides in groups:
        xs = x + offsets(v, (1 - OVERLAP) * MARK_SIZE * dx, MARK_SIZE * dy)
        # flagged sections keep their place in the fan but carry their own
        # marker, so they are identifiable without being removed
        plain = [i for i, s in enumerate(slides) if s not in MARKED]
        ax.plot(xs[plain], v[plain], linestyle='none', marker=marker,
                markersize=MARK_SIZE, markerfacecolor=colour,
                markeredgecolor='black', markeredgewidth=MARK_EDGE, zorder=3)
        for i, s in enumerate(slides):
            if s in MARKED:
                ax.plot([xs[i]], [v[i]], linestyle='none', marker=MARKED[s][0],
                        markersize=MARK_SIZE * 1.9, markerfacecolor=FLAG_GREY,
                        markeredgecolor='black', markeredgewidth=MARK_EDGE,
                        zorder=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--density', default=str(HERE / 'density.csv'))
    ap.add_argument('--out', default=str(HERE / 'density-by-depth.pdf'))
    ap.add_argument('--stats', default=str(HERE / 'density_statistics.csv'))
    ap.add_argument('--exclude', nargs='*', default=EXCLUDED,
                    help='sections to drop; pass with no names to keep all')
    a = ap.parse_args()

    data = load(a.density)
    everything = [s for s in sorted(data[BANDS[0]]['ground truth'])
                  if s not in a.exclude]
    if a.exclude:
        print('excluded: ' + ', '.join(a.exclude))
    # one pair of limits for both panels, so the species are read against the
    # same scale rather than each being re-zoomed to its own range
    kept = [data[b][arm][s] for b in BANDS for arm, *_ in ARMS for s in everything]
    lo, hi = min(kept), max(kept)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    rows, pending = [], []
    for ax, (title, belongs) in zip(axes, SPECIES):
        ax.set_facecolor(SURFACE)
        slides = [s for s in everything if belongs(s)]
        if not slides:
            raise SystemExit(f'no slides matched {title}')
        r, g = panel(ax, title, data, slides, lo, hi)
        rows += r
        pending.append((ax, g))

    fig.supylabel('MYL4 intensity normalised to total tissue intensity  $\\rho$',
                  fontsize=AXIS_TITLE, color=INK)
    fig.supxlabel('Depth into myocardium', fontsize=AXIS_TITLE, color=INK)
    axes[1].tick_params(left=False, labelleft=False)
    handles = [Line2D([], [], linestyle='none', marker=m, markersize=MARK_SIZE,
                      markerfacecolor=c, markeredgecolor='black',
                      markeredgewidth=MARK_EDGE) for _, _, c, m in ARMS]
    labels = [label for _, label, _, _ in ARMS]
    for marker, note in MARKED.values():
        handles.append(Line2D([], [], linestyle='none', marker=marker,
                              markersize=MARK_SIZE * 1.5, markerfacecolor=FLAG_GREY,
                              markeredgecolor='black', markeredgewidth=MARK_EDGE))
        labels.append(note)
    fig.legend(handles, labels, loc='upper center', ncol=len(handles),
               frameon=False, fontsize=LEGEND_TEXT, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.99))

    fig.tight_layout(rect=(0.01, 0.02, 1, 0.93))
    fig.canvas.draw()
    for ax, g in pending:
        draw_points(ax, g)

    out = Path(a.out).with_suffix('.pdf')
    fig.savefig(out, facecolor=SURFACE, bbox_inches='tight')
    print(f'wrote {out}')

    with open(a.stats, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {a.stats}\n')
    print(f"  {'species':<8}{'band':<12}{'n':>3}{'gt':>9}{'pred':>9}{'bias':>10}"
          f"{'95% LoA':>20}{'width':>9}{'mean|d|':>9}{'% effect':>10}")
    for r in rows:
        print(f"  {r['species']:<8}{r['band']:<12}{r['n']:>3}{r['gt_mean']:>9.4f}"
              f"{r['pred_mean']:>9.4f}{r['bias']:>+10.4f}"
              f"{r['loa_lower']:>+10.4f},{r['loa_upper']:>+9.4f}"
              f"{r['loa_width']:>9.4f}{r['mean_abs_difference']:>9.4f}"
              f"{r['pct_of_effect']:>9.0f}%")
    print(f"\n  Excluded: {', '.join(a.exclude) if a.exclude else 'none'}. "
          '% effect is the mean absolute difference against\n  '
          "the annotation's own near-minus-deep span: "
          + '  '.join(f"{r['species']} {r['effect']:.4f}"
                      for r in rows if r['band'] == BANDS[0]))


if __name__ == '__main__':
    main()
