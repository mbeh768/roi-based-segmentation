"""Relative MYL4 density by depth, annotation against every arm.

load() -> agreement() -> draw_*(). --arms picks the methods, so two arms gives
the annotation-versus-model figure and five the baseline comparison, from one
generator. >200 um is tabulated but not plotted: every arm lands within 0.003 of
the annotation there. AJKY121-8-2 is excluded, see EXCLUSIONS.md.

Theme follows refrence-figures-hillestad/pig_vs_human_PF_density_Fig1.R.

    python figures-for-pub/quantification/figure_density.py
    python figures-for-pub/quantification/figure_density.py --arms "ground truth" predicted
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
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent

# annotation and model keep their published colours; the baselines take distinct
# hues, ordered so adjacent boxes are never the closest pair
ARMS = (('ground truth',     'Ground truth',          '#2B8CBE', 'o'),
        ('predicted',        'Predicted',             '#FF7F0E', 's'),
        ('SAM',              'SAM',                   '#4C9F70', '^'),
        ('microSAM',         'microSAM',              '#8C6BB1', 'D'),
        ('ROI - foreground', 'ROI − FG', '#B3553B', 'v'))
GT = 'ground truth'
# keys match density.csv; rings() builds the top band as dist >= 200, so the
# label is an inequality the source string understates
BANDS = {'0-99 um': '0–99 µm', '100-199 um': '100–199 µm', '>200 um': '≥200 µm'}
PLOTTED = ['0-99 um', '100-199 um']
# specimen ids carry a capital P for pig; the human cases are AJK*
SPECIES = (('Pig', lambda s: 'P' in s), ('Human', lambda s: 'P' not in s))
EXCLUDED = ('AJKY121-8-2',)
LOA_Z = 1.96

INK, INK_2, INK_3 = '#111111', '#52514e', '#8a8984'
FRAME, GRID, SURFACE = '#222222', '#D9D9D9', '#FFFFFF'
BOX_EDGE, BOX_ALPHA = '#7A7A7A', 0.22
BORDER_LW, TICK_LW, BOX_LW = 0.75, 0.55, 0.60
MARK_SIZE, MARK_EDGE = 5.5, 0.75
AXIS_TEXT, AXIS_TITLE, LEGEND_TEXT = 15.0, 17.2, 20.0
PANEL_TITLE, TABLE_TEXT = 17.2, 17.2
DODGE, BOX_WIDTH, OVERLAP = 0.185, 0.155, 0.40

# fonttype 42 embeds TrueType rather than converting glyphs to paths
matplotlib.rcParams.update({'font.family': 'sans-serif', 'pdf.fonttype': 42,
                            'ps.fonttype': 42, 'svg.fonttype': 'none'})


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bland_altman(reference, method, z=LOA_Z):
    """bias +/- z*SD of (method - reference), in units of rho.

    Signed method-minus-annotation, so a positive bias is an overestimate. This
    matches gen_csv.py's `<arm>-GT` columns; the two disagreed before.

    loa_* is where an individual section falls and does not narrow with n.
    ci_* is the confidence interval on the bias itself; t rather than z
    because n is 8-11 here.
    """
    d = np.asarray(method, float) - np.asarray(reference, float)
    n = d.size
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    half = float(student_t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return {'bias': bias, 'sd_difference': sd,
            'loa_lower': bias - z * sd, 'loa_upper': bias + z * sd,
            'ci_lower': bias - half, 'ci_upper': bias + half,
            'mean_abs_difference': float(np.abs(d).mean())}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(path, arms):
    """{band: {arm: {slide: rho}}}, from quantify_density.py's CSV."""
    out = defaultdict(lambda: defaultdict(dict))
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['arm'] in arms:
                out[r['band']][r['arm']][r['slide']] = float(r['rho_px'])
    return out


def agreement(data, slides, species, arms):
    """One record per (species, band, arm), against the annotation."""
    out = []
    for band in BANDS:
        v = {a: np.array([data[band][a][s] for s in slides]) for a, *_ in arms}
        for arm, *_ in arms:
            if arm == GT:
                continue
            out.append({'species': species, 'band': band, 'arm': arm,
                        'n': len(slides), 'gt_mean': float(v[GT].mean()),
                        'arm_mean': float(v[arm].mean()),
                        **bland_altman(v[GT], v[arm])})
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def fan(values, span, tol):
    """Spread colliding points sideways about the centre. No jitter, so the
    figure is identical on every run."""
    out, group = np.zeros(len(values)), []
    for idx in np.argsort(values, kind='stable'):
        if group and values[idx] - values[group[0]] > tol:
            group = []
        k = len(group)
        out[idx] = span * ((k + 1) // 2) * (1 if k % 2 else -1) if k else 0.0
        group.append(idx)
    return out


def draw_boxes(ax, title, data, slides, ylim, arms):
    """One species panel. Returns the point sets, which can only be fanned once
    the axes has its final size."""
    groups = []
    for i, band in enumerate(PLOTTED):
        for j, (arm, _, colour, marker) in enumerate(arms):
            v = np.array([data[band][arm][s] for s in slides])
            x = i + (j - (len(arms) - 1) / 2) * DODGE
            # whiskers to the extremes: every point is drawn, so a 1.5 IQR
            # whisker would leave points stranded outside it
            bp = ax.boxplot([v], positions=[x], widths=BOX_WIDTH, whis=(0, 100),
                            showfliers=False, patch_artist=True, manage_ticks=False)
            for part in ('whiskers', 'caps', 'medians', 'boxes'):
                for artist in bp[part]:
                    artist.set(color=BOX_EDGE, linewidth=BOX_LW)
            bp['boxes'][0].set(facecolor=colour, alpha=BOX_ALPHA, edgecolor=BOX_EDGE)
            groups.append((x, v, colour, marker))

    ax.set_xticks(range(len(PLOTTED)))
    ax.set_xticklabels([BANDS[b] for b in PLOTTED], fontsize=AXIS_TITLE, color=INK)
    ax.tick_params(axis='x', length=0, pad=8)
    ax.tick_params(axis='y', labelsize=AXIS_TEXT, colors=INK, color=FRAME,
                   width=TICK_LW, length=5)
    ax.set_xlim(-0.62, len(PLOTTED) - 0.38)
    ax.set_ylim(*ylim)
    for sp in ax.spines.values():
        sp.set(visible=True, color=FRAME, linewidth=BORDER_LW)
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title(f'{title}  (n = {len(slides)})', fontsize=PANEL_TITLE, color=INK,
                 pad=10, loc='left', fontweight='bold')
    return groups


def draw_points(ax, groups):
    box = ax.get_window_extent()
    per_pt = ax.figure.dpi / 72.0
    dx = np.diff(ax.get_xlim())[0] / (box.width / per_pt)
    dy = np.diff(ax.get_ylim())[0] / (box.height / per_pt)
    for x, v, colour, marker in groups:
        ax.plot(x + fan(v, (1 - OVERLAP) * MARK_SIZE * dx, MARK_SIZE * dy), v,
                linestyle='none', marker=marker, markersize=MARK_SIZE,
                markerfacecolor=colour, markeredgecolor='black',
                markeredgewidth=MARK_EDGE, zorder=3)


def draw_table(ax, rows, arms):
    """Panel (c). Rows laid out from an index, so a group rule is the midpoint
    between two known row positions and cannot land on text."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('(c) Agreement with the annotation', fontsize=PANEL_TITLE,
                 color=INK, pad=10, loc='left', fontweight='bold')

    metrics = (('Bias [95% CI]',
                lambda r: f"{r['bias']:+.3f}  [{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}]"),
               ('LoA ±', lambda r: f"{LOA_Z * r['sd_difference']:.3f}"))
    compare = [a for a in arms if a[0] != GT]
    species = [s for s, _ in SPECIES]
    record = {(r['species'], r['band'], r['arm']): r for r in rows}
    n_of = {r['species']: r['n'] for r in rows}

    # normalised, while TABLE_TEXT is in points: retune if the font changes.
    # Column centres, with the spans they are centred in -- the bias cell is
    # 0.237 wide against a 0.136 header, so anchoring both on one edge would
    # leave the numbers off their title.
    X_BAND, X_ARM = 0.005, 0.175
    xs = [0.424, 0.599, 0.794, 0.969]
    span = ((0.306, 0.630), (0.676, 1.000))
    TOP, DY, GAP = 0.845, 0.058, 0.040
    right = span[-1][1]

    def y_of(i):
        return TOP - (i + 1) * DY - (i // len(compare)) * GAP

    for sp, (lo, hi) in zip(species, span):
        ax.text((lo + hi) / 2, TOP + 0.090, f'{sp}  (n = {n_of[sp]})',
                fontsize=TABLE_TEXT + 1.0, color=INK, ha='center', fontweight='bold')
        ax.plot([lo, hi], [TOP + 0.070] * 2, color=FRAME, lw=0.7, clip_on=False)
    for k, (label, _) in enumerate(metrics * len(species)):
        ax.text(xs[k], TOP + 0.016, label, fontsize=TABLE_TEXT, color=INK_2,
                ha='center')
    ax.plot([X_BAND, right], [TOP] * 2, color=FRAME, lw=0.7, clip_on=False)

    for bi, band in enumerate(BANDS):
        for ai, (arm, label, colour, marker) in enumerate(compare):
            y = y_of(bi * len(compare) + ai)
            if ai == 0:
                ax.text(X_BAND, y, BANDS[band], fontsize=TABLE_TEXT, color=INK,
                        fontweight='bold')
            ax.plot([X_ARM - 0.024], [y + 0.012], linestyle='none', marker=marker,
                    markersize=MARK_SIZE - 0.5, markerfacecolor=colour,
                    markeredgecolor='black', markeredgewidth=MARK_EDGE, clip_on=False)
            ax.text(X_ARM, y, label, fontsize=TABLE_TEXT, color=INK)
            for si, sp in enumerate(species):
                r = record[(sp, band, arm)]
                for k, (_, fmt) in enumerate(metrics):
                    ax.text(xs[si * len(metrics) + k], y, fmt(r),
                            fontsize=TABLE_TEXT, color=INK, ha='center')
        if bi + 1 < len(BANDS):
            last = y_of(bi * len(compare) + len(compare) - 1)
            nxt = y_of((bi + 1) * len(compare))
            ax.plot([X_BAND, right], [(last + nxt) / 2] * 2, color=GRID, lw=0.6,
                    clip_on=False)

    ax.text((X_BAND + right) / 2, y_of(len(BANDS) * len(compare) - 1) - 0.055,
            'bias = mean(arm − annotation)   ·   '
            '95% CI on the bias = bias ± t·SD/√n   ·   '
            '95% LoA = bias ± 1.96 SD',
            fontsize=TABLE_TEXT - 2.2, color=INK_3, ha='center')


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--density', default=str(HERE / 'density.csv'))
    ap.add_argument('--out', default=str(HERE / 'density-by-depth.pdf'))
    ap.add_argument('--stats', default=str(HERE / 'density_statistics.csv'))
    ap.add_argument('--arms', nargs='*', default=[a for a, *_ in ARMS])
    ap.add_argument('--exclude', nargs='*', default=EXCLUDED)
    a = ap.parse_args()

    arms = tuple(x for x in ARMS if x[0] in a.arms or x[0] == GT)
    data = load(a.density, {x[0] for x in arms})
    missing = [x[0] for x in arms if not data[list(BANDS)[0]].get(x[0])]
    if missing:
        raise SystemExit(f'{a.density} has no rows for: {", ".join(missing)}')
    slides = sorted(set.intersection(*[set(data[b][arm]) for b in BANDS
                                       for arm, *_ in arms]) - set(a.exclude))
    print(f'excluded: {", ".join(a.exclude) or "none"}   n = {len(slides)}   '
          f'arms: {", ".join(x[0] for x in arms)}')

    by_species = {name: [s for s in slides if belongs(s)] for name, belongs in SPECIES}
    rows = []
    for name, members in by_species.items():
        if not members:
            raise SystemExit(f'no slides matched {name}')
        rows += agreement(data, members, name, arms)

    # one pair of limits for both panels, so the species are read against the
    # same scale rather than each being re-zoomed to its own range
    plotted = [data[b][arm][s] for b in PLOTTED for arm, *_ in arms for s in slides]
    pad = 0.10 * (max(plotted) - min(plotted))
    ylim = (min(plotted) - pad, max(plotted) + pad)

    # row pitch in draw_table is normalised, so panel C's share of the figure
    # is what sets the physical spacing between rows
    fig = plt.figure(figsize=(12.2, 14.0))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.07)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axes[1].sharey(axes[0])

    pending = []
    for ax, (name, _) in zip(axes, SPECIES):
        ax.set_facecolor(SURFACE)
        pending.append((ax, draw_boxes(ax, f'({"ab"[len(pending)]}) {name}', data,
                                       by_species[name], ylim, arms)))
        ax.set_xlabel('Depth into myocardium', fontsize=AXIS_TITLE, color=INK,
                      labelpad=6)
    axes[1].tick_params(left=False, labelleft=False)
    axes[0].set_ylabel('Relative MYL4 intensity  $\\rho$',
                       fontsize=AXIS_TITLE, color=INK)

    fig.legend([Line2D([], [], linestyle='none', marker=m, markersize=MARK_SIZE + 1.5,
                       markerfacecolor=c, markeredgecolor='black',
                       markeredgewidth=MARK_EDGE) for _, _, c, m in arms],
               [l for _, l, _, _ in arms], loc='upper center', ncol=len(arms),
               frameon=False, fontsize=LEGEND_TEXT, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.985))

    table = fig.add_subplot(gs[1, :])
    draw_table(table, rows, arms)
    # explicit margins: panel C is positioned text, which tight_layout cannot
    # measure -- it warns and then guesses
    fig.subplots_adjust(left=0.085, right=0.985, top=0.900, bottom=0.025)
    # the table needs no room for a y axis label, so it starts left of the plots.
    # draw_table's widths are axes-normalised and assume this 0.970 span.
    box = table.get_position()
    table.set_position([0.015, box.y0, 0.970, box.height])
    fig.canvas.draw()
    for ax, groups in pending:
        draw_points(ax, groups)
    fig.savefig(a.out, facecolor=SURFACE, bbox_inches='tight')
    print(f'wrote {a.out}')

    with open(a.stats, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {a.stats}\n')
    print(f"  {'species':<8}{'band':<12}{'arm':<18}{'gt':>8}{'arm':>8}"
          f"{'bias':>9}{'95% CI':>20}{'LoA ±':>9}{'mean|d|':>9}")
    for r in rows:
        print(f"  {r['species']:<8}{r['band']:<12}{r['arm']:<18}{r['gt_mean']:>8.4f}"
              f"{r['arm_mean']:>8.4f}{r['bias']:>+9.4f}"
              f"{r['ci_lower']:>+10.4f},{r['ci_upper']:>+9.4f}"
              f"{LOA_Z * r['sd_difference']:>9.4f}{r['mean_abs_difference']:>9.4f}")


if __name__ == '__main__':
    main()
