"""density.csv -> one row per (band, slide): the annotation's rho and each arm
minus it. Every slide is listed; those the figure drops carry an asterisk.

    python figures-for-pub/quantification/gen_csv.py
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

GT = 'ground truth'
ARM_ORDER = [GT, 'predicted', 'SAM', 'microSAM', 'ROI - foreground']
BAND_ORDER = ['0-99 um', '100-199 um', '>200 um']
MARKED = ('AJKY121-8-2',)        # excluded from the figure; see EXCLUSIONS.md


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--density', default=str(HERE / 'density.csv'))
    ap.add_argument('--out', default=str(HERE / 'density_deltas.csv'))
    ap.add_argument('--value', default='rho_px')
    ap.add_argument('--precision', type=int, default=4)
    ap.add_argument('--mark', nargs='*', default=MARKED)
    a = ap.parse_args()

    rho = defaultdict(dict)
    for r in csv.DictReader(open(a.density)):
        rho[(r['band'], r['slide'])][r['arm']] = float(r[a.value])

    arms = [x for x in ARM_ORDER if any(x in v for v in rho.values())]
    others = [x for x in arms if x != GT]
    # multi-word arms are parenthesised so the subtraction reads unambiguously
    head = {x: (f'({x})-GT' if ' ' in x else f'{x}-GT') for x in others}
    fields = ['band', 'species', 'slide', 'GT'] + [head[x] for x in others]

    rows = []
    for band in BAND_ORDER:
        for (_, slide) in sorted((k for k in rho if k[0] == band),
                                 key=lambda k: ('P' not in k[1], k[1])):
            v = rho[(band, slide)]
            rows.append({'band': band, 'species': 'Pig' if 'P' in slide else 'Human',
                         'slide': slide + ('*' if slide in a.mark else ''),
                         'GT': f'{v[GT]:.{a.precision}f}',
                         **{head[x]: f'{v[x] - v[GT]:+.{a.precision}f}'
                            for x in others}})

    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {a.out}  ({len(rows)} rows)')

    width = {h: max(len(h), *(len(r[h]) for r in rows)) + 2 for h in fields}
    print('\n' + ''.join(h.rjust(width[h]) for h in fields))
    for r in rows:
        print(''.join(r[h].rjust(width[h]) for h in fields))


if __name__ == '__main__':
    main()
