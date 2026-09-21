"""Slide-level held-out IoU for a leave-one-case-out sweep.

This is the number the paper reports. It is NOT the test_iou in
test_results.json, and the two are not comparable:

  test_iou      a TILE metric. Each tile gets its own IoU, averaged over the
                held-out tiles. A tile is a 512px window, so this asks "how
                good is the average window" and never reconstructs the slide.

  slide IoU     one IoU per slide, over the stitched prediction, scored at the
                SOURCE TIFF's own resolution. This asks "how good is the
                segmentation of this specimen", which is what a reader of the
                paper cares about.

Ground truth comes from <stem>_mask.* in raw_image_directory, never from the
stitched tile masks. Scoring against the preprocessed label would compare two
arrays that went through the same resize, so part of the error cancels and the
number flatters itself; the claim is about the actual slide. --canvas-gt keeps
the old tile-mask behaviour for reproducing earlier JSONs only.

Two corrections are what make it slide-level rather than a re-average:

  overlap   preprocessing uses stride 256 with tile 512, so interior pixels are
            covered by 4 tiles. Summing per-tile intersections would count them
            up to four times. Predictions are accumulated onto an
            image_H x image_W canvas, DIVIDED BY THE COVERAGE COUNT, and
            thresholded once at the end.

  augment   train.py's held-out evaluation reuses the loader's dataset object,
            which has augment=True, so every test tile was seen under a random
            flip/rot90 draw. That is label-preserving and therefore unbiased,
            but it is noise on a reported number. Everything here runs with
            augmentation OFF.

Scope. Only tiles that survived roi_exclusion are forwarded, so the canvas
covers the ROI-adjacent region rather than the whole slide; covered_frac
records how much. Pixels no tile covered are excluded from both intersection
and union rather than counted as correct background, which would inflate the
score by the area of empty slide.

Usage:
    python utils/slide_level_iou.py --name k-fold-baseline \
        --outdir figures-for-pub/baseline
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch

# the repo root is the directory holding train.py, at whatever depth
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'train.py').exists())
sys.path.insert(0, str(ROOT))

import data_loader.data_loaders as module_data   # noqa: E402
import model.model as module_arch                # noqa: E402


def score_native(pred, covered, stem, raw_dir):
    """Score a stitched canvas against the SOURCE mask TIFF at its own resolution.

    preprocess.py resized native (H, W) so the longest side is `canvas` with aspect
    preserved, then zero-padded bottom-right to a square, so the inverse is exact:
    crop to (out_h, out_w) and resize back to (H, W). Nearest everywhere, because
    both are label masks and interpolation would invent boundary values.
    """
    hits = sorted(Path(raw_dir).glob(f'*/{stem}_mask.*'))
    if not hits:
        raise SystemExit(f'no source mask for {stem} under {raw_dir}')
    gt = tifffile.TiffFile(hits[0]).series[0].asarray()
    if gt.ndim == 3:
        gt = gt[..., 0] if gt.shape[-1] <= 4 else gt[0]
    gt = gt > 127                       # matches preprocess.py's >0.5 after /255
    H, W = gt.shape

    canvas = pred.shape[0]
    s = canvas / max(H, W)
    oh, ow = round(H * s), round(W * s)
    up = lambda m: cv2.resize(m[:oh, :ow].astype(np.uint8), (W, H),
                              interpolation=cv2.INTER_NEAREST) > 0
    cov_n = up(covered)
    pred_n = up(pred) & cov_n
    gt &= cov_n

    inter = int(np.count_nonzero(pred_n & gt))
    union = int(np.count_nonzero(pred_n | gt))
    out = {'iou': inter / union if union else float('nan'),
           'covered_frac': float(cov_n.mean()),
           'gt_px': int(np.count_nonzero(gt)),
           'pred_px': int(np.count_nonzero(pred_n)),
           'native_hw': [int(H), int(W)]}
    return inter, union, out


def stitch_fold(model, ds, idxs, device, amp, amp_dtype, batch_size,
                raw_dir=None):
    """Accumulate predictions onto per-slide canvases and score each slide."""
    from torch.utils.data.dataloader import default_collate

    canvas = {}
    with torch.no_grad():
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s + batch_size]
            batch = default_collate([ds[i] for i in chunk])
            *inputs, target = batch
            inputs = [t.to(device) for t in inputs]
            target = target.to(device)
            with torch.autocast('cuda', dtype=amp_dtype, enabled=amp):
                out = model(*inputs)
            prob = torch.sigmoid(out.float()).squeeze(1).cpu().numpy()
            gt = (target > 0.5).float().squeeze(1).cpu().numpy()

            for k, i in enumerate(chunk):
                _, stem, e = ds.items[i]
                if stem not in canvas:
                    H, W = e['image_H'], e['image_W']
                    canvas[stem] = {'p': np.zeros((H, W), np.float32),
                                    'g': np.zeros((H, W), np.float32),
                                    'n': np.zeros((H, W), np.uint16)}
                y, x, h, w = e['y'], e['x'], e['height'], e['width']
                c = canvas[stem]
                c['p'][y:y + h, x:x + w] += prob[k]
                c['g'][y:y + h, x:x + w] += gt[k]
                c['n'][y:y + h, x:x + w] += 1

    per_slide, inter_all, union_all = {}, 0, 0
    for stem, c in canvas.items():
        covered = c['n'] > 0
        count = np.maximum(c['n'], 1)
        # average the overlapping predictions, THEN threshold once
        pred = (c['p'] / count > 0.5) & covered
        if raw_dir is not None:
            inter, union, rec = score_native(pred, covered, stem, raw_dir)
        else:
            gt = (c['g'] / count > 0.5) & covered
            inter = int(np.count_nonzero(pred & gt))
            union = int(np.count_nonzero(pred | gt))
            rec = {'iou': inter / union if union else float('nan'),
                   'covered_frac': float(covered.mean()),
                   'gt_px': int(np.count_nonzero(gt)),
                   'pred_px': int(np.count_nonzero(pred))}
        inter_all += inter
        union_all += union
        per_slide[stem] = rec
    # case-level IoU pools the halves of a paired case onto one score, which is
    # the unit the split holds out
    case_iou = inter_all / union_all if union_all else float('nan')
    return case_iou, per_slide


def main():
    ap = argparse.ArgumentParser(description='slide-level held-out IoU per fold')
    ap.add_argument('--name', required=True,
                    help='run name stem, e.g. k-fold-baseline (folds are <name>-fNN)')
    ap.add_argument('--models-dir', default=str(ROOT / 'saved' / 'models'))
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--loader-module', default=None,
                    help='path to a module exposing register(), for runs whose config '
                         'names a loader class that does not live in '
                         'data_loader.data_loaders -- e.g. an ablation: '
                         'figures-for-pub/roi-ablation/noise_roi.py. The model must be '
                         'evaluated through the loader it was TRAINED with.')
    ap.add_argument('--canvas-gt', action='store_true',
                    help='score against the stitched TILE masks on the tiling canvas '
                         'instead of the source mask TIFF. Only for reproducing older '
                         'JSONs -- reported metrics are native-resolution.')
    ap.add_argument('--raw-dir',
                    default=json.loads((ROOT / 'config.json').read_text())
                    ['preprocess']['raw_image_directory'],
                    help='source TIFFs, holding <stem>_mask.* for native-resolution GT')
    ap.add_argument('--outdir', required=True,
                    help="experiment folder to write <name>_slide_iou.json into, "
                         "e.g. figures-for-pub/baseline. Required so results land "
                         "with the experiment that produced them rather than "
                         "beside this script.")
    a = ap.parse_args()

    runs = sorted(Path(a.models_dir).glob(f'{a.name}-f*'))
    runs = [d for d in runs if (d / 'model_best.pth').exists() and (d / 'config.json').exists()]
    if not runs:
        raise SystemExit(f'no runs with a checkpoint under {a.models_dir}/{a.name}-f*')
    if a.loader_module:
        import importlib.util
        p = Path(a.loader_module).resolve()
        sys.path.insert(0, str(p.parent))
        spec = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.register()
        print(f'registered loaders from {p.name}')

    device = torch.device(a.device)
    print(f'{len(runs)} folds\n')

    results = []
    for d in runs:
        cfg = json.loads((d / 'config.json').read_text())
        args = dict(cfg['data_loader']['args'])
        # deterministic evaluation: augmentation off, workers off, no shuffling
        args.update(hflip=0.0, vflip=0.0, rot90=0.0, shuffle=False, num_workers=0,
                    batch_size=a.batch_size)
        loader_cls = getattr(module_data, cfg['data_loader']['type'], None)
        if loader_cls is None:
            raise SystemExit(
                f"{d.name}'s config names data loader '{cfg['data_loader']['type']}', which "
                f"is not in data_loader.data_loaders. If it comes from an ablation, pass "
                f"--loader-module pointing at the module that registers it.")
        loader = loader_cls(**args)
        ds, idxs = loader.dataset, list(loader.test_indices)
        assert not ds.augment, 'augmentation still on -- the eval would be noisy'

        model = getattr(module_arch, cfg['arch']['type'])(**cfg['arch']['args']).to(device)
        state = torch.load(d / 'model_best.pth', map_location=device, weights_only=False)
        model.load_state_dict(state['state_dict'])
        model.eval()

        amp = bool(cfg['trainer'].get('amp', False))
        amp_dtype = getattr(torch, cfg['trainer'].get('amp_dtype', 'bfloat16'))
        case_iou, per_slide = stitch_fold(model, ds, idxs, device, amp, amp_dtype,
                                          a.batch_size,
                                          raw_dir=None if a.canvas_gt else a.raw_dir)

        tile_iou = json.loads((d / 'test_results.json').read_text()).get('test_iou') \
            if (d / 'test_results.json').exists() else None
        results.append({'run': d.name, 'case': loader.split_cases.get('test'),
                        'slide_iou_case': case_iou, 'per_slide': per_slide,
                        'tile_iou_reported': tile_iou, 'n_test_tiles': len(idxs),
                        'best_epoch': state.get('epoch')})
        cov = np.mean([v['covered_frac'] for v in per_slide.values()])
        print(f"  {d.name:<28} slide {case_iou:.4f}   tile {tile_iou if tile_iou is None else f'{tile_iou:.4f}'}"
              f"   covered {cov*100:.1f}%   {len(idxs)} tiles")
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    v = np.array([r['slide_iou_case'] for r in results])
    t = np.array([r['tile_iou_reported'] for r in results if r['tile_iou_reported'] is not None])
    print(f"\nslide-level IoU  mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}  "
          f"min {v.min():.4f}  max {v.max():.4f}  n={len(v)}")
    if len(t):
        print(f"tile-level IoU   mean {t.mean():.4f}  sd {t.std(ddof=1):.4f}  (for contrast; "
              f"different quantity, not comparable)")

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f'{a.name}_slide_iou.json'
    out.write_text(json.dumps({'name': a.name, 'folds': results}, indent=2))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
