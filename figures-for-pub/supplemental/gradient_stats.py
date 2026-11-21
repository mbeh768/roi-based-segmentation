"""Subendocardial MYL4 gradient from the existing quantification results.

g = rho(0-99 um) - rho(100-199 um) per slide, for the ground truth and for each
method. Reports the ground-truth gradient with a t-based 95% CI per species, and
each method's mean(g) / mean(g_GT) as the fraction of that gradient retained.

Read-only: the only thing written is this folder's outputs/gradient_stats.csv.

    python figures-for-pub/supplemental/gradient_stats.py
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'train.py').exists())
DENSITY = ROOT / 'figures-for-pub' / 'quantification' / 'density.csv'

VALUE = 'rho_px'                  # the per-pixel rho the paper reports
NEAR, FAR = '0-99 um', '100-199 um'
GT = 'ground truth'
METHODS = ['predicted', 'SAM', 'microSAM', 'ROI - foreground']
LABEL = {GT: 'Ground truth', 'predicted': 'Predicted', 'SAM': 'SAM',
         'microSAM': 'microSAM', 'ROI - foreground': 'ROI-FG'}
EXCLUDED = ('AJKY121-8-2',)       # see figures-for-pub/quantification/EXCLUSIONS.md


def species(slide):
    """Explicit rather than figure_density.py's `'P' in stem`; the two agree on
    all twenty stems, but this refuses an unknown id instead of calling it human."""
    if slide.startswith('AJK'):
        return 'Human'
    if slide[:2].isdigit() and slide[2] == 'P' or slide.startswith('FL_'):
        return 'Pig'
    raise ValueError(f'unknown species: {slide}')


def load(path, exclude):
    """(slide, arm) -> {band: rho}, for the two bands the gradient needs."""
    rho = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['slide'] in exclude or r['band'] not in (NEAR, FAR):
                continue
            rho.setdefault((r['slide'], r['arm']), {})[r['band']] = float(r[VALUE])
    return rho


def gradients(rho, arm):
    """slide -> g, for every slide carrying both bands under this arm."""
    out = {}
    for (slide, a), bands in rho.items():
        if a != arm:
            continue
        if set(bands) != {NEAR, FAR}:
            raise SystemExit(f'{slide} / {arm}: missing a band')
        out[slide] = bands[NEAR] - bands[FAR]
    return out


def summarise(g):
    """Mean with a two-sided 95% CI; t rather than z because n is 8-11 here."""
    a = np.asarray(g, float)
    n = a.size
    mean, sd = float(a.mean()), float(a.std(ddof=1))
    half = float(student_t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return {'n': n, 'mean_g': mean, 'sd_g': sd,
            'ci_lower': mean - half, 'ci_upper': mean + half}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--density', default=str(DENSITY))
    ap.add_argument('--out', default=str(HERE / 'outputs' / 'gradient_stats.csv'))
    ap.add_argument('--exclude', nargs='*', default=EXCLUDED)
    a = ap.parse_args()

    rho = load(a.density, set(a.exclude))
    g = {arm: gradients(rho, arm) for arm in [GT] + METHODS}
    slides = sorted(g[GT])
    for arm in METHODS:
        if set(g[arm]) != set(slides):
            raise SystemExit(f'{arm} covers different slides than the ground truth')
    print(f'{a.density}\nexcluded: {", ".join(a.exclude) or "none"}   '
          f'n = {len(slides)} slides   g = rho({NEAR}) - rho({FAR})\n')

    members = {sp: [s for s in slides if species(s) == sp] for sp in ('Pig', 'Human')}
    rows, gt_stat, retained = [], {}, {}
    for sp, mem in members.items():
        gt_stat[sp] = summarise([g[GT][s] for s in mem])
        rows.append({'species': sp, 'arm': LABEL[GT], **gt_stat[sp],
                     'retained_pct': 100.0})
        retained[sp] = {}
        for arm in METHODS:
            st = summarise([g[arm][s] for s in mem])
            retained[sp][arm] = 100.0 * st['mean_g'] / gt_stat[sp]['mean_g']
            rows.append({'species': sp, 'arm': LABEL[arm], **st,
                         'retained_pct': retained[sp][arm]})

    print(f"  {'species':<8}{'arm':<14}{'n':>3}{'mean g':>10}{'SD':>9}"
          f"{'95% CI':>20}{'retained':>10}")
    for r in rows:
        print(f"  {r['species']:<8}{r['arm']:<14}{r['n']:>3}{r['mean_g']:>+10.4f}"
              f"{r['sd_g']:>9.4f}{r['ci_lower']:>+10.4f},{r['ci_upper']:>+9.4f}"
              f"{r['retained_pct']:>9.1f}%")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {out}')

    others = {sp: max(retained[sp][m] for m in METHODS if m != 'predicted')
              for sp in members}
    p, h = gt_stat['Pig'], gt_stat['Human']
    sentence = (
        'The ground-truth gradient was '
        f"{p['mean_g']:+.3f} (95% CI {p['ci_lower']:.3f} to {p['ci_upper']:.3f}) "
        f"in porcine and {h['mean_g']:+.3f} ({h['ci_lower']:.3f} to "
        f"{h['ci_upper']:.3f}) in human sections; the proposed network retained "
        f"{retained['Pig']['predicted']:.0f}% and "
        f"{retained['Human']['predicted']:.0f}%, compared with at most "
        f"{others['Pig']:.0f}% and {others['Human']:.0f}% for the other methods.")
    print(f'\n  "{sentence}"')
    print(f"\n  R = max over {', '.join(LABEL[m] for m in METHODS if m != 'predicted')}"
          f"   Pig {others['Pig']:.1f}%   Human {others['Human']:.1f}%")


if __name__ == '__main__':
    main()
