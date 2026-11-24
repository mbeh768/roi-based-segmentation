"""Bland-Altman agreement with the ground truth, by species and depth band.

load() -> agreement() -> draw_ba(), plus write_tex() for the same numbers as a
booktabs table. 2x2 panels: species across, band down. >=200 um is tabulated but
not plotted -- every method lands within 0.003 of the ground truth there.
AJKY121-8-2 is excluded, see EXCLUSIONS.md.

Theme follows refrence-figures-hillestad/pig_vs_human_PF_density_Fig1.R.

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
from matplotlib.patches import Rectangle
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent

# Predicted takes the same blue the roi-upsample and noise-ablate figures use
ARMS = (('ground truth',     'Ground truth', '#2B8CBE', 'o'),
        ('predicted',        'Predicted',    '#2B8CBE', 'o'),
        ('SAM',              'SAM',          '#9A9A9A', 'o'),
        ('microSAM',         'μSAM',         '#9A9A9A', 'o'),
        ('ROI - foreground', 'ROI−FG',       '#9A9A9A', 'o'))
GT = 'ground truth'
# keys match density.csv; rings() builds the top band as dist >= 200, so the
# label is an inequality the source string understates
BANDS = {'0-99 um': '0–99µm', '100-199 um': '100–199µm', '>200 um': '≥200µm'}
TEX_BANDS = {'0-99 um': '0--99', '100-199 um': '100--199', '>200 um': r'$\geq$200'}
PLOTTED = ['0-99 um', '100-199 um']
# specimen ids carry a capital P for pig; the human cases are AJK*
SPECIES = (('Pig', lambda s: 'P' in s), ('Human', lambda s: 'P' not in s))
EXCLUDED = ('AJKY121-8-2',)
LOA_Z = 1.96

INK, INK_2, INK_3 = '#111111', '#52514e', '#8a8984'
FRAME, GRID, SURFACE = '#222222', '#D9D9D9', '#FFFFFF'
LOA_FILL = '#E8E8E8'
BORDER_LW, TICK_LW = 0.75, 0.55
MARK_SIZE, MARK_EDGE = 5.5, 0.75
# the figure is placed at 0.8\textwidth, so printed point size is ~0.41 of these
AXIS_TEXT, AXIS_TITLE, PANEL_TITLE = 20.0, 22.0, 22.0
# column geometry in x data units, one unit per method
LOA_W, BIAS_W, CAP_W, JITTER = 0.62, 0.30, 0.13, 0.055

# fonttype 42 embeds TrueType rather than converting glyphs to paths
matplotlib.rcParams.update({'font.family': 'sans-serif', 'pdf.fonttype': 42,
                            'ps.fonttype': 42, 'svg.fonttype': 'none'})


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bland_altman(reference, method, z=LOA_Z):
    """bias +/- z*SD of (method - reference), in units of rho.

    Signed method-minus-ground-truth, so a positive bias is an overestimate.
    ci_* is the confidence interval on the bias; t rather than z because n is
    8-11 here. loa_* is where an individual section falls and does not narrow
    with n.
    """
    d = np.asarray(method, float) - np.asarray(reference, float)
    n = d.size
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    half = float(student_t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return {'bias': bias, 'sd_difference': sd,
            'loa_lower': bias - z * sd, 'loa_upper': bias + z * sd,
            'ci_lower': bias - half, 'ci_upper': bias + half,
            'mean_abs_difference': float(np.abs(d).mean())}


def gradient_retained(data, slides, methods):
    """Each method's mean subendocardial gradient as a fraction of the ground
    truth's, in percent. g = rho(0-99) - rho(100-199) per slide, the same
    definition as figures-for-pub/supplemental/gradient_stats.py."""
    near, far = PLOTTED
    g = lambda arm: np.array([data[near][arm][s] - data[far][arm][s]
                              for s in slides]).mean()
    gt = g(GT)
    return {arm: 100.0 * g(arm) / gt for arm in methods}


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
    """One record per (species, band, arm), against the ground truth."""
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


def diffs(data, band, arm, slides):
    return np.array([data[band][arm][s] - data[band][GT][s] for s in slides])


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def fan(values, span, tol):
    """Spread colliding points sideways about the centre. No random jitter, so
    the figure is identical on every run."""
    out, group = np.zeros(len(values)), []
    for idx in np.argsort(values, kind='stable'):
        if group and values[idx] - values[group[0]] > tol:
            group = []
        k = len(group)
        out[idx] = span * ((k + 1) // 2) * (1 if k % 2 else -1) if k else 0.0
        group.append(idx)
    return out


def row_limits(data, by_species, band, methods):
    """One pair of limits per row, so the two species are read against the same
    scale. The LoA rectangles are included or the widest ones would clip."""
    vals = []
    for slides in by_species.values():
        for arm in methods:
            d = diffs(data, band, arm, slides)
            st = bland_altman(np.zeros_like(d), d)
            vals += [d.min(), d.max(), st['loa_lower'], st['loa_upper']]
    pad = 0.08 * (max(vals) - min(vals))
    return min(vals) - pad, max(vals) + pad


def draw_ba(ax, title, data, band, slides, methods, ylim):
    """One species x band panel of signed differences."""
    ax.set_facecolor(SURFACE)
    ax.axhline(0.0, color=FRAME, lw=0.9, zorder=2)

    for j, (arm, _, colour, marker) in enumerate(methods):
        d = diffs(data, band, arm, slides)
        st = bland_altman(np.zeros_like(d), d)

        ax.add_patch(Rectangle((j - LOA_W / 2, st['loa_lower']), LOA_W,
                               st['loa_upper'] - st['loa_lower'],
                               facecolor=LOA_FILL, edgecolor='none', zorder=0))
        ax.plot(j + fan(d, JITTER, 0.025 * (ylim[1] - ylim[0])), d,
                linestyle='none', marker=marker, markersize=MARK_SIZE,
                markerfacecolor=colour, markeredgecolor='black',
                markeredgewidth=MARK_EDGE, zorder=3)
        ax.plot([j - BIAS_W, j + BIAS_W], [st['bias']] * 2, color=INK,
                lw=2.2, solid_capstyle='butt', zorder=5)
        ax.plot([j, j], [st['ci_lower'], st['ci_upper']], color=INK, lw=1.3,
                zorder=5)
        for y in (st['ci_lower'], st['ci_upper']):
            ax.plot([j - CAP_W, j + CAP_W], [y] * 2, color=INK, lw=1.3, zorder=5)

    ax.set_xlim(-0.62, len(methods) - 0.38)
    ax.set_ylim(*ylim)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([l for _, l, _, _ in methods], fontsize=AXIS_TEXT, color=INK)
    ax.tick_params(axis='x', length=0, pad=8)
    ax.tick_params(axis='y', labelsize=AXIS_TEXT, colors=INK, color=FRAME,
                   width=TICK_LW, length=5)
    for sp in ax.spines.values():
        sp.set(visible=True, color=FRAME, linewidth=BORDER_LW)
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=PANEL_TITLE, color=INK, pad=10, loc='left',
                 fontweight='bold')


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def tex_label(label):
    """The figure labels carry Unicode glyphs pdflatex cannot set."""
    return label.replace('−', '$-$').replace('μ', r'$\mu$')


def num(v, sign=True):
    """Three decimals in math mode, so the sign is a LaTeX minus. Never -0.000,
    which reads as a typo once typeset -- but a dropped sign would shift the digits
    out of line with the signed rows, so a zero keeps the sign's width instead."""
    s = f'{v:+.3f}' if sign else f'{v:.3f}'
    if sign and float(s) == 0:
        s = r'\phantom{+}' + s.lstrip('+-')
    return f'${s}$'


def write_tex(path, rows, retained, methods, n_of, species):
    record = {(r['species'], r['band'], r['arm']): r for r in rows}
    head = (f'Method & ' + ' & '.join(r'Bias [95\% CI] & LoA $\pm$'
                                      for _ in species) + r' \\')
    out = [r'\begin{table}[htbp]', r'\centering',
           r'\caption{Bland--Altman agreement of each method with the ground truth '
           r'for relative MYL4 intensity $\rho_B$. Bias is the mean of '
           r'(method $-$ ground truth) with its 95\% confidence interval; limits '
           r'of agreement (LoA) are bias $\pm 1.96$~SD. Gradient retained is the '
           r'mean of $\rho_{0\text{--}99} - \rho_{100\text{--}199}$ for each '
           r'method as a percentage of that for the ground truth. Slide '
           r'AJKY121-8-2 is excluded.}',
           r'\label{tab:agreement}',
           # the bias cells carry an interval each, so the default column
           # padding overruns the text block
           r'\setlength{\tabcolsep}{4pt}\small',
           # space between the caption and the table; a bare \\ here has no line
           # to end, so LaTeX refuses it
           r'\medskip',
           r'\begin{tabular}{@{}l ll ll@{}}', r'\toprule',
           ' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{sp} (n = {n_of[sp]})}}'
                              for sp in species) + r' \\',
           r'\cmidrule(lr){2-3} \cmidrule(lr){4-5}', head, r'\midrule']

    for bi, band in enumerate(BANDS):
        if bi:
            out.append(r'\addlinespace')
        out.append(f'\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{TEX_BANDS[band]} '
                   r'\si{\micro\meter}}} \\')
        for arm, label, _, _ in methods:
            cells = []
            for sp in species:
                r = record[(sp, band, arm)]
                cells += [f"{num(r['bias'])} [{num(r['ci_lower'])}, "
                          f"{num(r['ci_upper'])}]",
                          num(LOA_Z * r['sd_difference'], sign=False)]
            out.append(' & '.join([tex_label(label)] + cells) + r' \\')

    out += [r'\addlinespace',
            r'\multicolumn{5}{@{}l}{\textit{Gradient retained (\%)}} \\']
    for arm, label, _, _ in methods:
        out.append(' & '.join([tex_label(label)] + [
            f'\\multicolumn{{2}}{{c}}{{${retained[sp][arm]:.0f}$}}'
            for sp in species]) + r' \\')

    out += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    Path(path).write_text('\n'.join(out) + '\n', encoding='utf-8')
    print(f'wrote {path}')


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--density', default=str(HERE / 'density.csv'))
    ap.add_argument('--out', default=str(HERE / 'density-by-depth.pdf'))
    ap.add_argument('--tex', default=str(HERE / 'agreement_table.tex'))
    ap.add_argument('--stats', default=str(HERE / 'density_statistics.csv'))
    ap.add_argument('--arms', nargs='*', default=[a for a, *_ in ARMS])
    ap.add_argument('--exclude', nargs='*', default=EXCLUDED)
    a = ap.parse_args()

    arms = tuple(x for x in ARMS if x[0] in a.arms or x[0] == GT)
    methods = tuple(x for x in arms if x[0] != GT)
    data = load(a.density, {x[0] for x in arms})
    missing = [x[0] for x in arms if not data[list(BANDS)[0]].get(x[0])]
    if missing:
        raise SystemExit(f'{a.density} has no rows for: {", ".join(missing)}')
    slides = sorted(set.intersection(*[set(data[b][arm]) for b in BANDS
                                       for arm, *_ in arms]) - set(a.exclude))
    print(f'excluded: {", ".join(a.exclude) or "none"}   n = {len(slides)}   '
          f'arms: {", ".join(x[0] for x in arms)}')

    by_species = {name: [s for s in slides if belongs(s)] for name, belongs in SPECIES}
    for name, members in by_species.items():
        if not members:
            raise SystemExit(f'no slides matched {name}')
    names = list(by_species)
    n_of = {sp: len(m) for sp, m in by_species.items()}

    rows = []
    for name, members in by_species.items():
        rows += agreement(data, members, name, arms)
    retained = {sp: gradient_retained(data, m, [x[0] for x in methods])
                for sp, m in by_species.items()}

    fig, axes = plt.subplots(len(PLOTTED), len(names), figsize=(12.2, 11.0))
    fig.patch.set_facecolor(SURFACE)
    for bi, band in enumerate(PLOTTED):
        ylim = row_limits(data, by_species, band, [x[0] for x in methods])
        for si, sp in enumerate(names):
            ax = axes[bi][si]
            letter = 'abcd'[bi * len(names) + si]
            draw_ba(ax, f'{letter}  {sp} (n = {n_of[sp]}) · {BANDS[band]}',
                    data, band, by_species[sp], methods, ylim)
            if si:
                ax.tick_params(left=False, labelleft=False)
            else:
                ax.set_ylabel('$\\rho_B$, method − ground truth',
                              fontsize=AXIS_TITLE, color=INK)

    fig.tight_layout(h_pad=2.4, w_pad=1.2)
    fig.savefig(a.out, facecolor=SURFACE, bbox_inches='tight')
    print(f'wrote {a.out}')

    write_tex(a.tex, rows, retained, methods, n_of, names)

    with open(a.stats, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {a.stats}\n')
    print(f"  {'species':<8}{'band':<12}{'arm':<18}{'bias':>9}{'95% CI':>20}"
          f"{'LoA ±':>9}")
    for r in rows:
        print(f"  {r['species']:<8}{r['band']:<12}{r['arm']:<18}"
              f"{r['bias']:>+9.4f}{r['ci_lower']:>+10.4f},{r['ci_upper']:>+9.4f}"
              f"{LOA_Z * r['sd_difference']:>9.4f}")
    print('\n  gradient retained (%)')
    for sp in names:
        print(f"    {sp:<6} " + '   '.join(
            f'{l} {retained[sp][arm]:.1f}' for arm, l, _, _ in methods))


if __name__ == '__main__':
    main()
