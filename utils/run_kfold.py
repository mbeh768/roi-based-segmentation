"""
Driver for leave-one-case-out cross validation: runs train.py once per fold,
sequentially, so every case is validated exactly once.

A "case" is a slide plus its paired twin half if it has one (see
data_loader.data_loaders.CASE_GROUPS) -- validating on one half while training
on its twin would leak adjacent tissue, so pairs move together. With
n_folds == number of cases, each fold holds out exactly one case, which is the
smallest possible validation set.

Run from the directory that `data_dir` and `save_dir` are relative to (the
repo's parent when data/ and saved/ live outside the repo):

    python att_vent_seg/utils/run_kfold.py \
        --repo att_vent_seg \
        --data-dir "data/[collapsed]t8192_ctx512_inner512_old" \
        --name "KFOLD_LOCO"

Folds that already have a completed run dir are skipped, so re-running after an
interrupt picks up where it left off. Aggregate the results afterwards with:

    python att_vent_seg/utils/kfold_report.py --name "KFOLD_LOCO"
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path


def discover_cases(repo, data_dir):
    """Ask the real dataset + CASE_GROUPS which cases exist, in a subprocess so
    the parent doesn't hold the dataset in memory for the whole run."""
    code = f'''
import sys, json
sys.path.insert(0, {str(repo)!r})
import data_loader.datasets as ds
from data_loader.data_loaders import case_id
d = ds.LocalGlobalDataset({str(data_dir)!r}, require_roi=True)
groups = {{}}
for _, stem, _ in d.items:
    groups.setdefault(case_id(stem), set()).add(stem)
print("@@@" + json.dumps({{c: sorted(s) for c, s in groups.items()}}))
'''
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    line = next((l for l in out.stdout.splitlines() if l.startswith('@@@')), None)
    if line is None:
        raise SystemExit(f"could not enumerate cases:\n{out.stdout}\n{out.stderr}")
    return json.loads(line[3:])


def completed_folds(save_dir, name):
    """Fold indices that already have a run dir with a finished model_best.pth."""
    done = {}
    models = Path(save_dir) / 'models'
    for cfg_path in sorted(models.glob('*/config.json')):
        try:
            cfg = json.loads(cfg_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if cfg.get('name') != name:
            continue
        args = cfg.get('data_loader', {}).get('args', {})
        if not args.get('n_folds'):
            continue
        if (cfg_path.parent / 'model_best.pth').exists():
            done[args.get('fold', 0)] = cfg_path.parent.name
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base-dir', default=None,
                    help='chdir here first -- config data_dir/save_dir are relative to it. '
                         'Needed when the caller (e.g. an MCP runner) fixes cwd elsewhere.')
    ap.add_argument('--repo', default='att_vent_seg', help='path to the repo checkout')
    ap.add_argument('--config', default=None, help='config path (default: <repo>/config.json)')
    ap.add_argument('--data-dir', required=True, help='preprocessed tensor dir, as train.py will see it')
    ap.add_argument('--name', required=True, help='experiment name shared by all folds')
    ap.add_argument('--save-dir', default='saved', help='where run dirs land (must match config trainer.save_dir)')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--early-stop', type=int, default=None)
    ap.add_argument('--folds', default=None, help='comma-separated subset, e.g. "0,3,7" (default: all)')
    ap.add_argument('--log-dir', default='kfold_logs', help='per-fold stdout/stderr capture')
    ap.add_argument('--dry-run', action='store_true', help='print the plan and the commands, run nothing')
    a = ap.parse_args()

    if a.base_dir:
        import os
        os.chdir(a.base_dir)
        print(f"cwd -> {os.getcwd()}")

    repo = Path(a.repo).resolve()
    config = Path(a.config) if a.config else repo / 'config.json'
    train_py = repo / 'train.py'
    for p in (repo, config, train_py):
        if not p.exists():
            raise SystemExit(f"missing: {p}")

    groups = discover_cases(repo, a.data_dir)
    cases = sorted(groups)
    n_folds = len(cases)
    n_slides = sum(len(v) for v in groups.values())

    # Mirror the loader's own ordering so the printed fold->case map is the real
    # one: it shuffles the case list with split_seed then np.array_split's it.
    import numpy as np
    cfg_args = json.loads(config.read_text())['data_loader']['args']
    split_seed = cfg_args.get('split_seed', 0) or 0
    n_val_cases = cfg_args.get('n_val_cases', 0) or 0
    shuffled = list(groups)
    np.random.RandomState(split_seed).shuffle(shuffled)
    chunks = [list(c) for c in np.array_split(np.array(shuffled, dtype=object), n_folds)]

    print(f"{n_slides} slides -> {n_folds} cases -> {n_folds} folds "
          f"(leave-one-case-out, split_seed={split_seed})")
    print(f"per fold: {n_folds - 1 - n_val_cases} train / {n_val_cases} val / 1 test cases\n")
    print(f"{'fold':>4}  {'TEST case':<18} {'val cases (model selection)':<46} test slides")
    for f, chunk in enumerate(chunks):
        rest = [c for c in shuffled if c not in set(chunk)]
        if n_val_cases:
            start = (f * n_val_cases) % len(rest)
            val = [rest[(start + i) % len(rest)] for i in range(n_val_cases)]
        else:
            val = list(chunk)
        for c in chunk:
            print(f"{f:>4}  {c:<18} {', '.join(val):<46} {', '.join(groups[c])}")

    wanted = ([int(x) for x in a.folds.split(',')] if a.folds else list(range(n_folds)))
    done = completed_folds(a.save_dir, a.name)
    if done:
        print(f"\nalready complete, skipping: {sorted(done)}")
    todo = [f for f in wanted if f not in done]
    print(f"\nto run: {todo}\n")

    started = time.time()
    results = {}
    for i, fold in enumerate(todo, 1):
        cmd = [sys.executable, '-u', str(train_py), '-c', str(config),
               '--n_folds', str(n_folds), '--fold', str(fold),
               '--data_dir', a.data_dir, '--name', a.name]
        if a.epochs is not None:
            cmd += ['--epochs', str(a.epochs)]
        if a.early_stop is not None:
            cmd += ['--early_stop', str(a.early_stop)]

        held_out = ', '.join(s for c in chunks[fold] for s in groups[c])
        print(f"=== [{i}/{len(todo)}] fold {fold} -- holding out {held_out} ===", flush=True)
        if a.dry_run:
            print('  ' + ' '.join(cmd))
            continue

        log_dir = Path(a.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{a.name.replace(' ', '_')}_fold{fold:02d}.log"
        # MKL_THREADING_LAYER: torch links libgomp, and mkl-service refuses to
        # load the Intel layer alongside it ("MKL_THREADING_LAYER=INTEL is
        # incompatible with libgomp.so.1"), aborting the run at import time.
        # expandable_segments: this config needs ~23 GB at batch_size=40, which
        # only fits alongside another job on the card once the allocator stops
        # stranding memory in fixed-size reserved segments.
        import os
        child_env = dict(os.environ,
                         MKL_THREADING_LAYER='GNU',
                         PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')

        t0 = time.time()
        with open(log_path, 'w') as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                env=child_env).returncode
        dt = time.time() - t0
        results[fold] = rc
        status = 'ok' if rc == 0 else f'FAILED rc={rc}'
        print(f"  {status} in {timedelta(seconds=int(dt))}  (log: {log_path})", flush=True)
        if rc != 0:
            print(f"  --- tail of {log_path} ---", flush=True)
            print('\n'.join(Path(log_path).read_text().splitlines()[-25:]), flush=True)

    if not a.dry_run:
        elapsed = timedelta(seconds=int(time.time() - started))
        bad = [f for f, rc in results.items() if rc != 0]
        print(f"\nran {len(results)} fold(s) in {elapsed}; "
              + (f"FAILED: {bad}" if bad else "all succeeded"))
        print(f"\nnow run:  python {repo / 'utils' / 'kfold_report.py'} "
              f"--name {a.name!r} --models-dir {a.save_dir}/models --log-dir {a.save_dir}/log --per-slide")


if __name__ == '__main__':
    main()
