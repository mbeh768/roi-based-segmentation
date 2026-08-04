"""Slide-level held-out IoU for the leave-one-case-out sweep.

The per-fold test_iou written by train.py is a *tile* metric: IoU computed per
batch and averaged over batches. That weights a batch of near-empty tiles the
same as a batch full of ventricle, and it never reconstructs the slide. This
recomputes the number the way a pathologist would read it -- one IoU per slide,
over the whole stitched prediction.

Two corrections matter here:

  overlap   preprocessing uses stride=256 with tile_size=512, so interior pixels
            are covered by 4 tiles. Summing per-tile intersections would count
            those pixels up to 4 times. Predictions are instead stitched onto an
            image_H x image_W canvas and averaged where they overlap, then
            thresholded once.

  augment   the test loader shares the dataset object with training, whose
            self.augment is True, so train.py's test pass saw a random
            flip/rot90 draw per tile. It is label-preserving (image and mask are
            transformed together) so it should not bias IoU, but it adds noise.
            Everything here runs with augmentation OFF, and tile_iou_batchmean
            is reported so the two can be compared directly.

Scope: only tiles that survived roi_exclusion are in the test set, so the canvas
covers the ROI-adjacent region rather than the entire slide. covered_frac
records how much of the canvas that is.

Usage:
    python utils/slide_level_iou.py --name KFOLD_LOCO \
        --models-dir saved/models --out slide_level_iou.json
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from utils import read_json, write_json


def crop_to_target(output, target):
    """Mirror model.metric._crop_to_target so numbers stay comparable."""
    if output.shape[-2:] == target.shape[-2:]:
        return output
    oh, ow = output.shape[-2:]
    th, tw = target.shape[-2:]
    y0, x0 = (oh - th) // 2, (ow - tw) // 2
    return output[..., y0:y0 + th, x0:x0 + tw]


def find_runs(models_dir, name):
    runs = {}
    for cfg_path in sorted(Path(models_dir).glob('*/config.json')):
        cfg = read_json(cfg_path)
        if cfg.get('name') != name:
            continue
        args = cfg['data_loader']['args']
        if not args.get('n_folds'):
            continue
        ckpt = cfg_path.parent / 'model_best.pth'
        if ckpt.exists():
            runs[args.get('fold', 0)] = (args.get('fold', 0), cfg, ckpt)
    return [runs[f] for f in sorted(runs)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True)
    ap.add_argument('--models-dir', default='saved/models')
    ap.add_argument('--data-dir', default=None, help='override config data_dir')
    ap.add_argument('--out', default='slide_level_iou.json')
    ap.add_argument('--batch-size', type=int, default=32)
    a = ap.parse_args()

    import data_loader.data_loaders as module_data
    import model.model as module_arch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    runs = find_runs(a.models_dir, a.name)
    if not runs:
        raise SystemExit(f"no completed fold runs for name={a.name!r} under {a.models_dir}")
    print(f"found {len(runs)} completed folds: {[r[0] for r in runs]}")

    results = {'folds': [], 'slides': [], 'name': a.name}

    for fold, cfg, ckpt in runs:
        args = dict(cfg['data_loader']['args'])
        if a.data_dir:
            args['data_dir'] = a.data_dir
        # deterministic evaluation: no geometric augmentation, no ROI jitter/dropout,
        # single-process so the ROI variant choice below is reproducible.
        args.update(hflip=0.0, vflip=0.0, rot90=0.0, roi_jitter=0.0,
                    roi_dropout=0.0, roi_distort=0.0,
                    shuffle=False, num_workers=0, fold=fold,
                    batch_size=a.batch_size)
        random.seed(1234)
        torch.manual_seed(1234)

        loader = getattr(module_data, cfg['data_loader']['type'])(**args)
        ds = loader.dataset
        test_idx = list(loader.test_indices)
        cases = loader.split_cases

        arch = cfg['arch']
        model = getattr(module_arch, arch['type'])(**arch['args']).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state['state_dict'])
        model.eval()
        use_amp = bool(cfg['trainer'].get('amp', False))
        amp_dtype = getattr(torch, cfg['trainer'].get('amp_dtype', 'bfloat16'))
        if cfg['trainer'].get('channels_last', False):
            model = model.to(memory_format=torch.channels_last)

        # canvases, one per stem in this fold's test case(s)
        canvas = {}
        for i in test_idx:
            _, stem, entry = ds.items[i]
            if stem not in canvas:
                H, W = entry['image_H'], entry['image_W']
                canvas[stem] = {'p': np.zeros((H, W), np.float32),
                                'g': np.zeros((H, W), np.float32),
                                'n': np.zeros((H, W), np.uint8)}

        tile_ious, inter_sum, union_sum = [], 0.0, 0.0
        with torch.no_grad():
            for s in range(0, len(test_idx), a.batch_size):
                chunk = test_idx[s:s + a.batch_size]
                batch = torch.utils.data.dataloader.default_collate(
                    [ds[i] for i in chunk])
                *raw, target, _g = batch
                inputs = []
                for x in raw:
                    x = x.to(device)
                    if x.dim() == 4 and cfg['trainer'].get('channels_last', False):
                        x = x.contiguous(memory_format=torch.channels_last)
                    inputs.append(x)
                target = target.to(device)
                with torch.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    out = model(*inputs)
                    if isinstance(out, tuple):
                        out = out[0]
                out = crop_to_target(out.float(), target)
                prob = torch.sigmoid(out)
                hard = (target > 0.5).float()

                # tile-level, matching model.metric.iou but with a hard target
                pb = (prob > 0.5).float()
                i_ = (pb * hard).sum()
                u_ = pb.sum() + hard.sum() - i_
                tile_ious.append(float((i_ + 1e-6) / (u_ + 1e-6)))
                inter_sum += float(i_)
                union_sum += float(u_)

                pn = prob.squeeze(1).cpu().numpy()
                gn = hard.squeeze(1).cpu().numpy()
                for k, idx in enumerate(chunk):
                    _, stem, entry = ds.items[idx]
                    y, x = entry['y'], entry['x']
                    h, w = entry['height'], entry['width']
                    c = canvas[stem]
                    c['p'][y:y + h, x:x + w] += pn[k]
                    c['g'][y:y + h, x:x + w] += gn[k]
                    c['n'][y:y + h, x:x + w] += 1

        # stitch -> one IoU per slide
        fold_i = fold_u = 0.0
        for stem, c in sorted(canvas.items()):
            cov = c['n'] > 0
            cnt = np.maximum(c['n'], 1)
            pred = (c['p'] / cnt > 0.5) & cov
            gt = (c['g'] / cnt > 0.5) & cov
            inter = float(np.count_nonzero(pred & gt))
            union = float(np.count_nonzero(pred | gt))
            iou = (inter + 1e-6) / (union + 1e-6)
            fold_i += inter
            fold_u += union
            results['slides'].append({
                'stem': stem, 'fold': fold,
                'case': next((k for k in cases['test']), None),
                'slide_iou': iou,
                'gt_px': int(np.count_nonzero(gt)),
                'pred_px': int(np.count_nonzero(pred)),
                'covered_frac': float(cov.mean()),
            })
            print(f"  fold {fold:>2}  {stem:<22} slide_iou={iou:.4f}  "
                  f"covered={cov.mean()*100:.1f}%")

        results['folds'].append({
            'fold': fold,
            'test_cases': cases['test'],
            'stems': sorted(canvas),
            'n_test_tiles': len(test_idx),
            'slide_iou_case': (fold_i + 1e-6) / (fold_u + 1e-6),
            'tile_iou_batchmean': float(np.mean(tile_ious)) if tile_ious else float('nan'),
            'tile_iou_pooled': (inter_sum + 1e-6) / (union_sum + 1e-6),
        })
        del model, canvas
        torch.cuda.empty_cache()

    fo = results['folds']
    for key in ('slide_iou_case', 'tile_iou_batchmean', 'tile_iou_pooled'):
        v = [f[key] for f in fo]
        results[f'mean_{key}'] = float(np.mean(v))
        results[f'std_{key}'] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0

    write_json(results, a.out)
    print(f"\n{'fold':>4}  {'case':<26} {'slide_IoU':>10} {'tile_mean':>10} {'tile_pooled':>12}")
    for f in fo:
        print(f"{f['fold']:>4}  {', '.join(f['test_cases']):<26} "
              f"{f['slide_iou_case']:>10.4f} {f['tile_iou_batchmean']:>10.4f} "
              f"{f['tile_iou_pooled']:>12.4f}")
    print(f"\n  slide-level IoU : {results['mean_slide_iou_case']:.4f} "
          f"+/- {results['std_slide_iou_case']:.4f}")
    print(f"  tile-level (mean): {results['mean_tile_iou_batchmean']:.4f} "
          f"+/- {results['std_tile_iou_batchmean']:.4f}")
    print(f"\nwrote {a.out}")


if __name__ == '__main__':
    main()
