"""Dump the SAM point prompts, so a figure can draw the points the model got.

export_sam_masks.py keeps only `n_points` per slide, not the coordinates. This
calls its own prompt_points() on the same roi_<stem>.pt, so the dump cannot drift
from what was prompted -- no reimplementation here, and no GPU: the points are
connected components and a lattice, not model output.

SAM and microSAM share the prompt, so one dump serves both. stub-roi uses no
points at all.

Coordinates are in SAM's SIDE frame, exactly as passed to predict(). For the
8192 canvas the agreement figures draw on, multiply by canvas_long / sam_side.

    python figures-for-pub/quantification/export_prompt_points.py
"""
import argparse
import json
from pathlib import Path

import export_sam_masks as esm

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cfg = json.loads((esm.ROOT / 'config.json').read_text())
    ap.add_argument('--data-dir',
                    default=str(esm.ROOT / cfg['data_loader']['args']['data_dir']))
    ap.add_argument('--point-spacing', type=int, default=16)
    ap.add_argument('--scores', default=str(HERE / 'sam_scores.json'),
                    help='cross-checked against its recorded n_points')
    ap.add_argument('--out', default=str(HERE / 'prompt_points.json'))
    ap.add_argument('--slides', nargs='*', default=None)
    a = ap.parse_args()

    scores = json.loads(Path(a.scores).read_text())
    want = {r['stem']: r['n_points'] for r in scores['sam']['slides']}
    stems = a.slides or sorted(want)

    out, bad = {}, []
    for stem in stems:
        roi = esm.load_roi(a.data_dir, stem)
        pts = esm.prompt_points(roi, a.point_spacing)
        out[stem] = {'n_points': len(pts),
                     'points': [[round(float(x), 2), round(float(y), 2)]
                                for x, y in pts]}
        flag = ''
        if stem in want and want[stem] != len(pts):
            flag = f'  !! sam_scores.json says {want[stem]}'
            bad.append(stem)
        print(f'  {stem:22} {len(pts):>4} points{flag}')

    Path(a.out).write_text(json.dumps(
        {'point_spacing': a.point_spacing, 'sam_side': esm.SIDE,
         'roi_side': esm.ROI_SIDE, 'arms': ['sam', 'microsam'],
         'slides': out}, indent=1) + '\n', encoding='utf-8')
    print(f'wrote {a.out}  ({len(out)} slides)')
    if bad:
        raise SystemExit('count differs from what was prompted: ' + ', '.join(bad))


if __name__ == '__main__':
    main()
