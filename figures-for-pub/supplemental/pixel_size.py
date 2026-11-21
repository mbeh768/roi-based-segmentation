"""Effective pixel size at the 8192^2 working resolution.

preprocess.py resizes native (H, W) so the longest side is CANVAS with aspect
preserved, then zero-pads bottom-right to a square (see utils/slide_level_metrics.py's
to_native), so one working pixel spans native_um_per_px * native_longest / CANVAS.

Two repo-held sources, both keyed by slide stem:
  figures-for-pub/quantification/microns_per_pixel.json  manually entered um/px
  figures-for-pub/model-baseline/<run>_slide_metrics.json  native_hw = GT mask shape

The um/px entries are the scale of the level actually loaded, not of level 0: the
5.114 slides were read from the 2x-downsampled pyramid level, and quantify_density.py's
rings() applies the same value to the same native_hw array.

Read-only: the only thing written is this folder's outputs/pixel_size.csv.

    python figures-for-pub/supplemental/pixel_size.py
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'train.py').exists())
QUANT = ROOT / 'figures-for-pub' / 'quantification'
BASELINE = ROOT / 'figures-for-pub' / 'model-baseline'

CANVAS = 8192


def species(slide):
    if slide.startswith('AJK'):
        return 'Human'
    if slide[:2].isdigit() and slide[2] == 'P' or slide.startswith('FL_'):
        return 'Pig'
    raise ValueError(f'unknown species: {slide}')


def native_sides(path):
    """stem -> (H, W), pooled over folds; a stem held out once appears once."""
    folds = json.loads(Path(path).read_text())['folds']
    out = {}
    for f in folds:
        for stem, v in f['per_slide'].items():
            hw = tuple(int(x) for x in v['native_hw'])
            if out.setdefault(stem, hw) != hw:
                raise SystemExit(f'{stem}: native_hw differs between folds')
    return out


def spread(values):
    return {'min': min(values), 'median': median(values), 'max': max(values)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--voxel', default=str(QUANT / 'microns_per_pixel.json'))
    ap.add_argument('--metrics',
                    default=str(BASELINE / 'k-fold-bnaug-2026-10-17_slide_metrics.json'))
    ap.add_argument('--canvas', type=int, default=CANVAS)
    ap.add_argument('--out', default=str(HERE / 'outputs' / 'pixel_size.csv'))
    a = ap.parse_args()

    voxel = json.loads(Path(a.voxel).read_text())
    hw = native_sides(a.metrics)
    if set(voxel) != set(hw):
        raise SystemExit('stems differ between the two sources: '
                         f'{sorted(set(voxel) ^ set(hw))}')

    rows = []
    for stem in sorted(hw, key=lambda s: (species(s) != 'Pig', s)):
        h, w = hw[stem]
        long_side = max(h, w)
        um = float(voxel[stem]['um_per_px'])
        rows.append({'slide': stem, 'species': species(stem),
                     'native_h': h, 'native_w': w, 'native_longest_px': long_side,
                     'native_um_per_px': um,
                     'scale_factor': long_side / a.canvas,
                     'effective_um_per_px': um * long_side / a.canvas,
                     'physical_longest_mm': um * long_side / 1000.0,
                     'upsampled': long_side < a.canvas})
    print(f'{a.voxel}\n{a.metrics}\ncanvas = {a.canvas}   n = {len(rows)}\n')

    print(f"  {'slide':<20}{'species':<8}{'native H x W':>15}{'longest':>9}"
          f"{'um/px':>8}{'factor':>8}{'eff um/px':>11}{'mm':>8}")
    for r in rows:
        print(f"  {r['slide']:<20}{r['species']:<8}"
              f"{r['native_h']:>7} x{r['native_w']:>6}{r['native_longest_px']:>9}"
              f"{r['native_um_per_px']:>8.3f}{r['scale_factor']:>8.3f}"
              f"{r['effective_um_per_px']:>11.3f}{r['physical_longest_mm']:>8.1f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0]))
        w_.writeheader()
        w_.writerows(rows)
    print(f'\nwrote {out}\n')

    eff = [r['effective_um_per_px'] for r in rows]
    groups = [('all', eff)] + [
        (sp, [r['effective_um_per_px'] for r in rows if r['species'] == sp])
        for sp in ('Pig', 'Human')]
    print(f"  {'group':<8}{'n':>3}{'min':>9}{'median':>9}{'max':>9}  effective um/px")
    for name, vals in groups:
        s = spread(vals)
        print(f"  {name:<8}{len(vals):>3}{s['min']:>9.3f}{s['median']:>9.3f}"
              f"{s['max']:>9.3f}")

    up = [r['slide'] for r in rows if r['upsampled']]
    print(f"\n  upsampled (longest side < {a.canvas}): "
          f"{', '.join(up) if up else 'none'}")


if __name__ == '__main__':
    main()
