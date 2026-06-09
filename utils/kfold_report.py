"""
Post-hoc k-fold report. Reads the run dirs produced by the EXPERIMENT 9L folds
(train.py ... --fold 0..k-1) and aggregates them — nothing is tracked inside the
training loop; this only parses what training already wrote.

Two modes:

  per-fold (default)  parse each fold's info.log, pick the best epoch by the
                      monitored metric (min val_loss), and report that epoch's
                      val_iou / val_loss across folds as mean +/- std. This is
                      the number that goes in the paper.

  --per-slide         load each fold's model_best.pth and evaluate its held-out
                      slides (augmentation OFF), reporting per-tile IoU averaged
                      per slide. Because k-fold validates every slide exactly
                      once, concatenating folds gives a full per-slide difficulty
                      table — answers "are the same slides always easy/hard?".

Usage:
    python utils/kfold_report.py --name "EXPERIMENT 9L"
    python utils/kfold_report.py --name "EXPERIMENT 9L" --per-slide
"""
import argparse
import re
import statistics
from pathlib import Path

from utils import read_json

# info.log lines look like:
#   2026-05-25 16:51:01,790 - trainer - INFO -     val_loss       : 0.5970902264118194
_KV = re.compile(r'-\s+(\w+)\s+:\s+([-+0-9.eE]+)\s*$')


def _parse_epochs(info_log):
    """Yield per-epoch dicts of {metric: float} parsed from an info.log."""
    epochs, cur = [], {}
    for line in Path(info_log).read_text().splitlines():
        m = _KV.search(line)
        if not m:
            continue
        key, val = m.group(1), float(m.group(2))
        if key == 'epoch':
            if cur:
                epochs.append(cur)
            cur = {}
        else:
            cur[key] = val
    if cur:
        epochs.append(cur)
    return epochs


def _best_epoch(epochs, mnt_metric='val_loss', mnt_mode='min'):
    """The epoch dict that optimizes the monitored metric (matches model_best.pth)."""
    have = [e for e in epochs if mnt_metric in e]
    if not have:
        return None
    return (min if mnt_mode == 'min' else max)(have, key=lambda e: e[mnt_metric])


def _find_fold_runs(models_dir, log_dir, name):
    """Return [(fold, config, info_log, ckpt)] for every run whose config name
    matches `name`, keyed by the fold index recorded in its saved config."""
    runs = {}
    for cfg_path in sorted(Path(models_dir).glob('*/config.json')):
        cfg = read_json(cfg_path)
        if cfg.get('name') != name:
            continue
        args = cfg['data_loader']['args']
        if not args.get('n_folds', 0):
            continue
        run = cfg_path.parent.name              # e.g. "EXPERIMENT 9L_2"
        info_log = Path(log_dir) / run / 'info.log'
        ckpt = cfg_path.parent / 'model_best.pth'
        fold = args.get('fold', 0)
        if fold in runs:
            print(f"  WARNING: two runs report fold={fold} "
                  f"({runs[fold][2].parent.name} and {run}); keeping the newer one")
        runs[fold] = (fold, cfg, info_log, ckpt)
    return [runs[f] for f in sorted(runs)]


def report_per_fold(runs, mnt_metric='val_loss', mnt_mode='min'):
    print(f"\n=== per-fold best ({mnt_mode} {mnt_metric}) ===")
    print(f"{'fold':>4}  {'epoch':>6}  {'val_loss':>10}  {'val_iou':>10}")
    ious, losses = [], []
    expected = runs[0][1]['data_loader']['args']['n_folds'] if runs else 0
    for fold, _cfg, info_log, _ckpt in runs:
        if not info_log.exists():
            print(f"{fold:>4}  {'-- missing info.log --':>30}")
            continue
        epochs = _parse_epochs(info_log)
        best = _best_epoch(epochs, mnt_metric, mnt_mode)
        if best is None:
            print(f"{fold:>4}  {'-- no val metrics --':>30}")
            continue
        # best epoch number = its 1-based position among parsed epoch blocks
        ep = epochs.index(best) + 1
        vi, vl = best.get('val_iou', float('nan')), best.get('val_loss', float('nan'))
        print(f"{fold:>4}  {ep:>6}  {vl:>10.4f}  {vi:>10.4f}")
        if vi == vi:      # not NaN
            ious.append(vi)
        if vl == vl:
            losses.append(vl)

    n = len(ious)
    if expected and n < expected:
        print(f"  NOTE: found {n}/{expected} folds — run the missing --fold values "
              f"before trusting the aggregate.")
    if n:
        def ms(xs):
            return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)
        mi, si = ms(ious)
        ml, sl = ms(losses) if losses else (float('nan'), 0.0)
        print(f"\n  val_iou : {mi:.4f} +/- {si:.4f}   (min {min(ious):.4f}, max {max(ious):.4f}, n={n})")
        print(f"  val_loss: {ml:.4f} +/- {sl:.4f}")
        print(f"\n  --> report: val IoU {mi:.3f} +/- {si:.3f} ({n}-fold CV)")
    return ious


def report_per_slide(runs):
    """Load each fold's model_best.pth and score its held-out slides (aug off).
    Uses LocalGlobalDataLoader itself to pick the fold's val indices, so the
    split is byte-identical to training. Per-tile IoU is averaged per stem,
    matching how val_iou is defined during training."""
    import torch
    from torch.utils.data.dataloader import default_collate
    import data_loader.data_loaders as module_data
    import model.model as module_arch
    import model.metric as module_metric

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []   # (stem, fold, iou, n_tiles)
    print("\n=== per-slide IoU (each slide held out exactly once) ===")
    for fold, cfg_dict, _info_log, ckpt in runs:
        if not ckpt.exists():
            print(f"  fold {fold}: missing {ckpt} — skipping")
            continue

        # Rebuild the loader for this fold with augmentation OFF so per-slide
        # numbers are deterministic; the fold split itself is seed-fixed.
        args = dict(cfg_dict['data_loader']['args'])
        args.update(hflip=0.0, vflip=0.0, rot90=0.0, shuffle=False, fold=fold)
        loader = getattr(module_data, cfg_dict['data_loader']['type'])(**args)
        ds = loader.dataset
        val_idx = list(loader.valid_sampler.indices)

        # instantiate the model directly (avoid ConfigParser — its constructor
        # creates fresh run directories as a side effect)
        arch = cfg_dict['arch']
        model = getattr(module_arch, arch['type'])(**arch['args']).to(device)
        state = torch.load(ckpt, map_location=device)['state_dict']
        model.load_state_dict(state)
        model.eval()
        iou = module_metric.iou

        # group this fold's val tiles by slide
        by_stem = {}
        for i in val_idx:
            by_stem.setdefault(ds.items[i][1], []).append(i)

        with torch.no_grad():
            for stem, idxs in sorted(by_stem.items()):
                vals = []
                for s in range(0, len(idxs), 32):       # mini-batch the tiles
                    batch = default_collate([ds[i] for i in idxs[s:s + 32]])
                    *raw, target, _gmask = batch        # drop unused global_mask
                    inputs = [x.to(device) for x in raw]
                    out = model(*inputs)
                    if isinstance(out, tuple):
                        out = out[0]
                    vals.append(float(iou(out, target.to(device))))
                slide_iou = sum(vals) / len(vals)
                rows.append((stem, fold, slide_iou, len(idxs)))

    rows.sort(key=lambda r: r[2])                       # hardest slides first
    print(f"{'slide':<28} {'fold':>4} {'tiles':>6} {'iou':>8}")
    for stem, fold, iou_v, n in rows:
        print(f"{stem:<28} {fold:>4} {n:>6} {iou_v:>8.4f}")
    if rows:
        import statistics as st
        vals = [r[2] for r in rows]
        print(f"\n  slides={len(rows)}  mean={st.mean(vals):.4f}  "
              f"std={st.stdev(vals) if len(vals) > 1 else 0.0:.4f}  "
              f"min={min(vals):.4f}  max={max(vals):.4f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True, help='config "name" shared by the fold runs')
    ap.add_argument('--models-dir', default='saved/models')
    ap.add_argument('--log-dir', default='saved/log')
    ap.add_argument('--per-slide', action='store_true',
                    help='also load each fold checkpoint and score its held-out slides')
    a = ap.parse_args()

    runs = _find_fold_runs(a.models_dir, a.log_dir, a.name)
    if not runs:
        raise SystemExit(f"No fold runs found for name='{a.name}' under {a.models_dir}")
    print(f"found {len(runs)} fold run(s): folds {[r[0] for r in runs]}")
    report_per_fold(runs)
    if a.per_slide:
        report_per_slide(runs)
