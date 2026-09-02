"""Driver for the ROI ablation, 14-fold leave-one-case-out.

The arm is the flagship LOCO run with one tensor swapped: the ROI handed to the
model is Gaussian noise instead of the annotation. The architecture is untouched,
so parameter count and every code path are identical -- the ROI channel is still
there, it just carries nothing.

    arm            roi_exclusion   tile selection        what the ROI still does
    -----------------------------------------------------------------------------
    noise_guided   True            the real ROI picks    chooses which tiles exist
    noise          False           every tile is used    nothing at all

noise_guided is the arm that runs. It isolates the ROI's role as an INPUT PRIOR:
the same tiles the flagship trained on, in the same order, at the same steps per
epoch, but the network can no longer see where the structure is. Because the tile
set is held fixed, the comparison against the flagship is clean -- one tensor is
the only difference between them.

`noise` additionally drops tile selection, so the ROI contributes nothing at all.
It is left defined but is NOT run by default: at tiling 8192 the ROI keeps 3997
of 19220 tiles, so that arm trains on ~4.8x the tiles per epoch (74k vs 16k
optimiser steps per fold at the flagship's ~215 epochs) for a result that mostly
restates what tile selection already implies -- the discarded tiles contain no
ventricle at all, so removing the filter buries the signal in guaranteed-empty
background. Run it with --arms noise if that changes.

The baseline is the existing flagship LOCO run, which is `noise_guided` with a
real ROI. Compare against its TILE-level test IoU (0.9190 +- 0.0759 across the
14 folds), which is what test_results.json records here; the 0.9243 headline is
the slide-level stitched number and is not like-for-like.

Usage:
    python roi-ablation/train_ablation.py                    # noise_guided, 14 folds
    python roi-ablation/train_ablation.py --folds 0,1,2
    python roi-ablation/train_ablation.py --arms noise       # the dropped arm
    python roi-ablation/train_ablation.py --dry-run          # write configs only
    python roi-ablation/train_ablation.py --collect          # summarise, run nothing

Completed folds are skipped on re-invocation (a fold counts as done only when it
has written test_results.json, so an interrupted run is retried rather than
silently treated as finished).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

ARMS = {
    # arm name      -> data_loader arg overrides
    'noise_guided': {'roi_exclusion': True,  'roi_exclusion_radius': 0.0},
    'noise':        {'roi_exclusion': False},
}


def build_config(base, arm, fold, args):
    cfg = json.loads(json.dumps(base))          # deep copy
    cfg['name'] = f'{args.prefix}_{arm}_f{fold:02d}'

    dl = cfg['data_loader']
    dl['type'] = 'NoiseROIDataLoader'
    dl['args'].update(ARMS[arm])
    dl['args'].update(n_folds=14, fold=fold, require_roi=True)
    if args.data_dir:
        dl['args']['data_dir'] = args.data_dir
    if args.batch_size:
        dl['args']['batch_size'] = args.batch_size
    if args.num_workers is not None:
        dl['args']['num_workers'] = args.num_workers

    # roi_exclusion=False leaves the radius unused; drop it so the saved config
    # does not imply a filter that never ran.
    if not dl['args'].get('roi_exclusion'):
        dl['args'].pop('roi_exclusion_radius', None)

    tr = cfg['trainer']
    if args.epochs:
        tr['epochs'] = args.epochs
    if args.early_stop:
        tr['early_stop'] = args.early_stop
    if args.accum_steps:
        tr['accum_steps'] = args.accum_steps
    return cfg


def run_dirs(save_root, name):
    """Every directory parse_config may have used for this run name.

    parse_config appends _1, _2, ... on collision, so a retried fold leaves
    several. Sorted so the newest-numbered comes last.
    """
    base = save_root / 'models'
    return sorted(base.glob(f'{name}') ) + sorted(base.glob(f'{name}_[0-9]*'))


def completed(save_root, name):
    for d in run_dirs(save_root, name):
        if (d / 'test_results.json').exists():
            return d
    return None


def main():
    ap = argparse.ArgumentParser(description='ROI ablation: both arms, 14-fold LOCO')
    ap.add_argument('--base-config', default=str(ROOT / 'kfold_results' / 'KFOLD_LOCO' / 'config.json'),
                    help='flagship LOCO config the arms are derived from')
    ap.add_argument('--arms', default='noise_guided',
                    help=f'subset of {list(ARMS)}; "noise" drops tile selection too '
                         f'and costs ~4.8x the steps per fold, so it is not run by default')
    ap.add_argument('--folds', default='0-13', help='e.g. 0-13 or 0,3,7')
    ap.add_argument('--prefix', default='ROI_ABL')
    ap.add_argument('--data-dir', default='data/[collapsed]t8192_ctx512_inner512',
                    help="overrides the '_old' copy the flagship config names; the two "
                         'are byte-identical (38581 files match on path, size and checksum)')
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--accum-steps', type=int, default=None,
                    help='micro-batching, for GPUs smaller than the one the flagship used')
    ap.add_argument('--num-workers', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--early-stop', type=int, default=None)
    ap.add_argument('--device', default=None, help='CUDA_VISIBLE_DEVICES value, e.g. "0"')
    ap.add_argument('--probe-every', type=int, default=1)
    ap.add_argument('--probe-hist-every', type=int, default=50)
    ap.add_argument('--dry-run', action='store_true', help='write configs, run nothing')
    ap.add_argument('--collect', action='store_true', help='summarise finished folds only')
    args = ap.parse_args()

    arms = [a for a in args.arms.split(',') if a]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f'unknown arm {a!r}; choose from {list(ARMS)}')
    if '-' in args.folds:
        lo, hi = args.folds.split('-')
        folds = list(range(int(lo), int(hi) + 1))
    else:
        folds = [int(x) for x in args.folds.split(',') if x != '']

    base = json.loads(Path(args.base_config).read_text())
    save_root = ROOT / base['trainer']['save_dir']
    cfg_dir = HERE / 'configs'
    cfg_dir.mkdir(exist_ok=True)
    (HERE / 'results').mkdir(exist_ok=True)

    # train.py resolves data_dir and save_dir relative to the CWD, so the
    # subprocess must run from the repo root regardless of where this was called.
    env = dict(os.environ)
    env.setdefault('MKL_THREADING_LAYER', 'GNU')
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    if args.device is not None:
        env['CUDA_VISIBLE_DEVICES'] = args.device

    jobs = [(arm, f) for arm in arms for f in folds]
    print(f'{len(jobs)} fold-runs: arms={arms} folds={folds[0]}..{folds[-1]}')
    t0 = time.time()

    for arm, fold in jobs:
        cfg = build_config(base, arm, fold, args)
        name = cfg['name']
        done = completed(save_root, name)
        if done is not None:
            print(f'  [skip] {name} -- already has {done.name}/test_results.json')
            continue
        cfg_path = cfg_dir / f'{name}.json'
        cfg_path.write_text(json.dumps(cfg, indent=4))
        if args.dry_run or args.collect:
            print(f'  [dry ] {name} -> {cfg_path.relative_to(ROOT)}')
            continue

        cmd = [sys.executable, str(HERE / '_train_one.py'), '-c', str(cfg_path),
               '--probe-every', str(args.probe_every),
               '--probe-hist-every', str(args.probe_hist_every)]
        print(f'\n=== {name} ===\n  {" ".join(cmd)}', flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT), env=env)
        if r.returncode != 0:
            print(f'  *** {name} exited {r.returncode}; continuing to next fold',
                  file=sys.stderr, flush=True)

    # ---- summary ------------------------------------------------------------
    summary = {}
    for arm in arms:
        rows = []
        for fold in folds:
            name = f'{args.prefix}_{arm}_f{fold:02d}'
            d = completed(save_root, name)
            if d is None:
                continue
            res = json.loads((d / 'test_results.json').read_text())
            res.update(fold=fold, arm=arm, run_dir=str(d.relative_to(ROOT)))
            rows.append(res)
        summary[arm] = rows

    out = HERE / 'results' / 'ablation_summary.json'
    out.write_text(json.dumps(summary, indent=2))

    print(f'\n{"arm":<15}{"folds":>6}{"test IoU":>10}{"sd":>8}{"best epoch":>12}')
    for arm, rows in summary.items():
        if not rows:
            print(f'{arm:<15}{0:>6}{"--":>10}')
            continue
        v = np.array([r['test_iou'] for r in rows])
        e = np.array([r.get('best_epoch') or 0 for r in rows])
        print(f'{arm:<15}{len(rows):>6}{v.mean():>10.4f}{v.std():>8.4f}{e.mean():>12.1f}')
    print(f'\nwrote {out.relative_to(ROOT)}   ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
