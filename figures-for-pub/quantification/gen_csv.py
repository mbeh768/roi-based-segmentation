"""density.csv -> one row per (band, slide): the annotation's rho and each arm
minus it. Slides the figure excludes carry an asterisk.

    python figures-for-pub/quantification/gen_csv.py
    python figures-for-pub/quantification/gen_csv.py --tex density_deltas_table.tex
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
GT = 'ground truth'
ARMS = [GT, 'predicted', 'SAM', 'microSAM', 'ROI - foreground']
SHORT = {'predicted': 'Predicted', 'SAM': 'SAM', 'microSAM': 'microSAM',
         'ROI - foreground': r'ROI$-$FG'}
# rings() builds the top band as dist >= 200, so the source label understates it
BANDS = {'0-99 um': '0-99 um', '100-199 um': '100-199 um', '>200 um': '>=200 um'}
MARKED = ('AJKY121-8-2',)        # excluded from the figure; see EXCLUSIONS.md
# the delivered filename drops the V that its paired half carries
RENAME = {'AJK056-10-2of2': 'AJKV056-10-2of2'}


def species(slide):
    """figure_density.py splits on `'P' in stem`; both agree on all twenty
    stems, but this refuses an unknown id rather than calling it human."""
    if slide.startswith('AJK'):
        return 'Human'
    if re.match(r'(\d{2}P|FL_)', slide):
        return 'Pig'
    raise ValueError(f'unknown species: {slide}')


def natkey(s):
    """21PControl-2 before 21PControl-11; '_' folds to '-' or _4 sorts last."""
    return [int(t) if t.isdigit() else t
            for t in re.split(r'(\d+)', s.replace('_', '-'))]


def fmt(d):
    """Signed, but never -0.0000, which reads as a typo once typeset."""
    s = f'{d:+.4f}'
    return '0.0000' if float(s) == 0 else s


def tex_band(label):
    """Order matters: '>=' must go before the bare '>', or it leaves a stray =."""
    s = label.replace(' um', r'~\si{\micro\metre}')
    s = s.replace('>=', r'$\geq$').replace('>', r'$>$')
    return s.replace('-', '--')


def write_tex(path, recs, others):
    esc = lambda s: s.replace('_', r'\_')
    n = 3 + len(others)
    head = ('\\toprule\n'
            f' & & & \\multicolumn{{{len(others)}}}{{c}}{{Difference from GT}} \\\\\n'
            f'\\cmidrule(l){{4-{n}}}\n'
            '{Slide} & {Species} & {GT} & '
            + ' & '.join('{' + SHORT[x] + '}' for x in others) + ' \\\\\n'
            '\\midrule')
    out = [r'\begingroup',
           r'\linespread{1}\selectfont\small',
           r'\setlength{\tabcolsep}{4pt}',
           r'\begin{longtable}{@{}ll S[table-format=1.4] '
           f'*{{{len(others)}}}{{S[table-format=+1.4, retain-explicit-plus]}}@{{}}}}',
           r'\caption{Extended quantification results per slide and depth band. GT is '
           r'the ground-truth density; the remaining columns give each method'
           r"'s difference from GT. A slide excluded from the main analysis is "
           r'marked with a star (*).}'
           r'\label{tab:density-deltas} \\',
           head, r'\endfirsthead',
           f'\\multicolumn{{{n}}}{{@{{}}l}}{{\\textit{{Table \\thetable\\ continued}}}} \\\\',
           head, r'\endhead', r'\midrule',
           f'\\multicolumn{{{n}}}{{r@{{}}}}{{\\textit{{Continued on next page}}}} \\\\',
           r'\endfoot', r'\bottomrule', r'\endlastfoot']

    band = None
    for r in recs:
        if r['band'] != band:
            if band is not None:
                out.append(r'\addlinespace')
            band = r['band']
            out.append(f'\\multicolumn{{{n}}}{{@{{}}l}}{{\\textit{{{tex_band(band)}}}}} \\\\')
        name = esc(r['slide']) + (r'\textsuperscript{*}' if r['excluded'] else '')
        out.append(' & '.join([name, r['species'], r['GT']]
                              + [r[x] for x in others]) + r' \\')
    out += [r'\end{longtable}', r'\endgroup']
    Path(path).write_text('\n'.join(out) + '\n', encoding='utf-8')
    print(f'wrote {path}  ({len(recs)} rows)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--density', default=str(HERE / 'density.csv'))
    ap.add_argument('--out', default=str(HERE / 'density_deltas.csv'))
    ap.add_argument('--tex', default=None, help='also write a longtable there')
    ap.add_argument('--value', default='rho_px')
    ap.add_argument('--mark', nargs='*', default=MARKED)
    a = ap.parse_args()

    rho = defaultdict(dict)
    with open(a.density, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            slide = RENAME.get(r['slide'], r['slide'])
            rho[(r['band'], slide)][r['arm']] = float(r[a.value])
    assert {b for b, _ in rho} <= set(BANDS), 'unknown band in density.csv'

    others = [x for x in ARMS[1:] if any(x in v for v in rho.values())]
    recs = [{'band': BANDS[b], 'species': species(s), 'slide': s,
             'excluded': s in a.mark, 'GT': f'{rho[(b, s)][GT]:.4f}',
             **{x: fmt(rho[(b, s)][x] - rho[(b, s)][GT]) for x in others}}
            for b in BANDS
            for _, s in sorted((k for k in rho if k[0] == b),
                               key=lambda k: (species(k[1]) != 'Pig', natkey(k[1])))]

    # multi-word arms are parenthesised so the subtraction reads unambiguously
    head = {x: (f'({x})-GT' if ' ' in x else f'{x}-GT') for x in others}
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['band', 'species', 'slide', 'GT']
                           + [head[x] for x in others])
        w.writeheader()
        for r in recs:
            w.writerow({'band': r['band'], 'species': r['species'],
                        'slide': r['slide'] + ('*' if r['excluded'] else ''),
                        'GT': r['GT'], **{head[x]: r[x] for x in others}})
    print(f'wrote {a.out}  ({len(recs)} rows)')

    if a.tex:
        write_tex(a.tex, recs, others)


if __name__ == '__main__':
    main()
