"""EXPERIMENT 15: does the segmentation hold when a different annotator draws the ROI?

Pure inference on EXPERIMENT 11's existing checkpoints. Fold k's model is only ever
applied to fold k's own held-out case, so every measurement comes from a model that
never saw that slide. Nothing is trained and no checkpoint is written.

THE METRIC is slide-level IoU vs ground truth of the final segmentation -- the same
measure EXPERIMENT 11 reports, so the y-axis is directly comparable to its 0.9243.
Perturbation magnitude is recorded only to label the x-axis.

Two different baselines, and they are NOT the same number:

  strength 0 here  the fixed no-suffix ROI, i.e. one reference annotator. This is the
                   sweep's own baseline and what every perturbed point is measured against.
  EXPERIMENT 11    datasets.py picks random.choice(stem_to_roi[stem]) per tile, and that
                   glob covers roi_<stem>.pt, _1 and _2 -- so 11's 0.9243 is already an
                   expectation over the THREE REAL annotated variants. It is itself a
                   (small-sample) measurement of real annotator variation.

The harness is therefore checked against 11 by replicating 11's protocol -- letting the
loader sample ROIs as it normally does -- rather than by comparing the pinned-ROI number
to it. That isolates "is the stitching correct" from "is the ROI protocol different".

Annotators are 5 FIXED presets, not random draws. _roi_grid_distort multiplies its
control points by `strength`, so one fixed seed gives one consistent annotator bias
whose magnitude scales with strength. Every strength is then a PAIRED comparison
against the same five annotators, which removes draw-to-draw variance from the trend.

ROI source is roi_<stem>.pt, the on-file annotation with no suffix. An ROI is NEVER
derived from the ground-truth mask: it is a user-supplied prior consumed at inference,
so building it from GT would leak the label into the model's input.

Usage:
    python utils/roi_robustness_sweep.py --models-dir <dir> --data-dir <dir> \
        --out roi_robustness.json
"""
import argparse, json, random, time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from data_loader.datasets import _roi_grid_distort

PRESET_SEED = 9100          # annotator p is always seed PRESET_SEED + p
SCALE = 32                  # ROI/global 256 px across an 8192 tiling space


def make_roi(base, strength, preset):
    """Annotator `preset` at `strength`. Same seed -> same warp field, scaled."""
    if strength == 0:
        return base.clone()
    torch.manual_seed(PRESET_SEED + preset)
    random.seed(PRESET_SEED + preset)
    return _roi_grid_distort(base.clone(), strength=strength)


def iou_np(a, b):
    return (np.count_nonzero(a & b) + 1e-9) / (np.count_nonzero(a | b) + 1e-9)


def surf_dist(a, b):
    a, b = (a > 0.5).numpy(), (b > 0.5).numpy()
    if not a.any() or not b.any():
        return float('nan')
    ea = a ^ ndimage.binary_erosion(a)
    eb = b ^ ndimage.binary_erosion(b)
    if not ea.any() or not eb.any():
        return float('nan')
    return float((ndimage.distance_transform_edt(~ea)[eb].mean() +
                  ndimage.distance_transform_edt(~eb)[ea].mean()) / 2)


def crop_to(o, t):
    if o.shape[-2:] == t.shape[-2:]:
        return o
    oh, ow = o.shape[-2:]; th, tw = t.shape[-2:]
    return o[..., (oh - th) // 2:(oh - th) // 2 + th, (ow - tw) // 2:(ow - tw) // 2 + tw]


def stitch_iou(model, ds, idxs, roi_by_stem, device, amp, dt, bs=32):
    """Stitch predictions onto the slide canvas and return per-case IoU vs GT.

    Overlapping tiles are averaged before a single threshold -- stride 256 with
    512px tiles covers interior pixels four times, so summing per-tile results
    would count them repeatedly. Same procedure as utils/slide_level_iou.py.
    """
    from torch.utils.data.dataloader import default_collate
    canv = {}
    with torch.no_grad():
        for s in range(0, len(idxs), bs):
            chunk = idxs[s:s + bs]
            batch = default_collate([ds[i] for i in chunk])
            *raw, target, _g = batch
            raw = [x.to(device) for x in raw]
            if roi_by_stem is not None:
                # override the loader's ROI with this annotator's version; None leaves the
                # loader's own random-variant choice in place (EXPERIMENT 11's protocol)
                raw[3] = torch.stack([roi_by_stem[ds.items[i][1]] for i in chunk]).to(device)
            target = target.to(device)
            with torch.autocast('cuda', dtype=dt, enabled=amp):
                out = model(*raw)
                if isinstance(out, tuple):
                    out = out[0]
            out = crop_to(out.float(), target)
            p = torch.sigmoid(out).squeeze(1).cpu().numpy()
            gt = (target > 0.5).float().squeeze(1).cpu().numpy()
            for k, i in enumerate(chunk):
                _, stem, e = ds.items[i]
                if stem not in canv:
                    H, W = e['image_H'], e['image_W']
                    canv[stem] = {'p': np.zeros((H, W), np.float32),
                                  'g': np.zeros((H, W), np.float32),
                                  'n': np.zeros((H, W), np.uint8)}
                y, x, h, w = e['y'], e['x'], e['height'], e['width']
                c = canv[stem]
                c['p'][y:y + h, x:x + w] += p[k]
                c['g'][y:y + h, x:x + w] += gt[k]
                c['n'][y:y + h, x:x + w] += 1
    inter = union = 0.0
    per_slide = {}
    for stem, c in canv.items():
        cov = c['n'] > 0
        cnt = np.maximum(c['n'], 1)
        pred = (c['p'] / cnt > 0.5) & cov
        gt = (c['g'] / cnt > 0.5) & cov
        i_ = np.count_nonzero(pred & gt); u_ = np.count_nonzero(pred | gt)
        inter += i_; union += u_
        per_slide[stem] = (i_ + 1e-9) / (u_ + 1e-9)
    return (inter + 1e-9) / (union + 1e-9), per_slide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-dir', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--name', default='KFOLD_LOCO')
    ap.add_argument('--strengths', default='0,0.005,0.01,0.02,0.04,0.08')
    ap.add_argument('--presets', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--reference', default='kfold_results/slide_level_iou.json',
                    help='EXPERIMENT 11 result, to assert strength 0 reproduces it')
    ap.add_argument('--degenerate', action='store_true',
                    help='evaluate ROIs carrying no or wrong localisation: all-ones '
                         '(full image), all-zeros (empty), and another slide\'s ROI')
    ap.add_argument('--real-variants', action='store_true',
                    help='pin each REAL annotated ROI variant in turn instead of warping; '
                         'measures actual inter-annotator effect on the same footing')
    ap.add_argument('--harness-check', action='store_true',
                    help="also re-run EXPERIMENT 11's own ROI protocol to verify stitching")
    ap.add_argument('--out', default='roi_robustness.json')
    a = ap.parse_args()

    import data_loader.data_loaders as module_data
    import model.model as module_arch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    S = [float(x) for x in a.strengths.split(',')]

    runs = {}
    for c in sorted(Path(a.models_dir).glob('*/config.json')):
        d = json.loads(c.read_text())
        if d.get('name') != a.name:
            continue
        ck = c.parent / 'model_best.pth'
        if d['data_loader']['args'].get('n_folds') and ck.exists():
            runs[d['data_loader']['args'].get('fold', 0)] = (d, ck)
    if not runs:
        raise SystemExit(f"no checkpoints for {a.name} under {a.models_dir}")
    print(f"{len(runs)} folds, {len(S)} strengths, {a.presets} presets "
          f"-> {len(runs)*len(S)*a.presets} evaluations (inference only)\n")

    all_stems = sorted(p.name for p in (Path(a.data_dir)/'images').iterdir() if p.is_dir())
    results = []
    t0 = time.time()
    for fold in sorted(runs):
        cfg, ckpt = runs[fold]
        args = dict(cfg['data_loader']['args'])
        args.update(data_dir=a.data_dir, hflip=0.0, vflip=0.0, rot90=0.0,
                    roi_jitter=0.0, roi_dropout=0.0, roi_distort=0.0,
                    shuffle=False, num_workers=0, fold=fold, batch_size=a.batch_size)
        random.seed(0); torch.manual_seed(0)
        loader = getattr(module_data, cfg['data_loader']['type'])(**args)
        ds, idxs = loader.dataset, list(loader.test_indices)
        stems = sorted({ds.items[i][1] for i in idxs})

        arch = cfg['arch']
        model = getattr(module_arch, arch['type'])(**arch['args']).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=False)['state_dict'])
        model.eval()
        amp = bool(cfg['trainer'].get('amp', False))
        dt = getattr(torch, cfg['trainer'].get('amp_dtype', 'bfloat16'))

        base = {s: torch.load(Path(a.data_dir) / 'images' / s / f'roi_{s}.pt',
                              weights_only=True) for s in stems}

        if a.degenerate:
            # The reviewer's question: how much of the result is the ROI itself?
            #   full   all-ones -- ROI present but carries ZERO localisation. If this still
            #          beats a no-ROI model, the mechanism helps beyond mere localisation,
            #          which is the strongest form of the result.
            #   empty  all-zeros -- the prior actively says "nothing here".
            #   wrong  another slide's ROI, resampled -- localisation present but WRONG,
            #          which separates "needs a prior" from "needs a CORRECT prior".
            other = sorted(set(all_stems) - set(stems))
            modes = {
                'full':  {s: torch.ones_like(base[s]) for s in stems},
                'empty': {s: torch.zeros_like(base[s]) for s in stems},
            }
            if other:
                donor = other[fold % len(other)]
                dr = torch.load(Path(a.data_dir)/'images'/donor/f'roi_{donor}.pt',
                                weights_only=True)
                modes['wrong'] = {s: torch.nn.functional.interpolate(
                    dr.unsqueeze(0), size=base[s].shape[-2:], mode='nearest')[0]
                    for s in stems}
            for mode, rois in modes.items():
                iou, per_slide = stitch_iou(model, ds, idxs, rois, device, amp, dt,
                                            a.batch_size)
                results.append({'fold': fold, 'stems': stems, 'strength': mode,
                                'preset': 0, 'slide_iou': iou, 'per_slide': per_slide,
                                'roi_iou': float('nan'), 'roi_surf_px': float('nan'),
                                'n_test_tiles': len(idxs)})
                print(f"  fold {fold:>2}  DEGENERATE {mode:<6} IoU {iou:.4f}", flush=True)
            del model; torch.cuda.empty_cache()
            continue

        if a.real_variants:
            # Each slide's genuinely annotated variants, pinned one at a time. Same
            # evaluation path as the synthetic sweep, so the two are directly comparable:
            # displacement magnitude is matched but the disagreement is REAL, i.e. still
            # anatomy-following rather than a warp off the structure.
            nv = min(len(ds.stem_to_roi[s]) for s in stems)
            for vi in range(nv):
                rois = {s: torch.load(sorted(ds.stem_to_roi[s])[vi], weights_only=True)
                        for s in stems}
                iou, per_slide = stitch_iou(model, ds, idxs, rois, device, amp, dt,
                                            a.batch_size)
                roi_iou = float(np.mean([iou_np((base[s][0] > 0.5).numpy(),
                                                (rois[s][0] > 0.5).numpy()) for s in stems]))
                sd = float(np.mean([surf_dist(base[s][0], rois[s][0]) for s in stems]))
                results.append({'fold': fold, 'stems': stems, 'strength': 'real',
                                'preset': vi, 'slide_iou': iou, 'per_slide': per_slide,
                                'roi_iou': roi_iou, 'roi_surf_px': sd * SCALE,
                                'n_test_tiles': len(idxs)})
                print(f"  fold {fold:>2}  REAL variant {vi}  IoU {iou:.4f}   "
                      f"(vs ref ROI: IoU {roi_iou:.3f}, {sd*SCALE:.0f} px)", flush=True)
            del model; torch.cuda.empty_cache()
            continue

        for strength in S:
            n_p = 1 if strength == 0 else a.presets      # strength 0 is one deterministic point
            for preset in range(n_p):
                rois = {s: make_roi(base[s], strength, preset) for s in stems}
                iou, per_slide = stitch_iou(model, ds, idxs, rois, device, amp, dt,
                                            a.batch_size)
                roi_iou = float(np.mean([iou_np((base[s][0] > 0.5).numpy(),
                                                (rois[s][0] > 0.5).numpy()) for s in stems]))
                sd = float(np.mean([surf_dist(base[s][0], rois[s][0]) for s in stems]))
                results.append({'fold': fold, 'stems': stems, 'strength': strength,
                                'preset': preset, 'slide_iou': iou,
                                'per_slide': per_slide, 'roi_iou': roi_iou,
                                'roi_surf_px': sd * SCALE,
                                'n_test_tiles': len(idxs)})
                print(f"  fold {fold:>2}  s={strength:<6g} preset {preset}  "
                      f"IoU {iou:.4f}   (ROI IoU {roi_iou:.3f}, {sd*SCALE:.0f} px)", flush=True)
        del model
        torch.cuda.empty_cache()

    ref = Path(a.reference)
    if ref.exists():
        r11 = {f['fold']: f['slide_iou_case'] for f in json.loads(ref.read_text())['folds']}
        d = [abs(r['slide_iou'] - r11[r['fold']])
             for r in results if r['strength'] == 0 and r['fold'] in r11]
        if d:   # only the synthetic sweep has a strength-0 point to compare
            print(f"\npinned-ROI baseline vs EXPERIMENT 11: mean |diff| {np.mean(d):.4f}, "
                  f"max {np.max(d):.4f}")
            print("  (expected to differ: 11 samples randomly among the 3 REAL annotated ROI "
                  "variants per tile,\n   so its 0.9243 already averages over real annotators; "
                  "this sweep pins the no-suffix ROI)")
        if a.harness_check:
            print("\nharness check -- replicating EXPERIMENT 11's own ROI protocol:")
            worst = 0.0
            for fold in sorted(runs):
                cfg, ckpt = runs[fold]
                ar = dict(cfg['data_loader']['args'])
                ar.update(data_dir=a.data_dir, hflip=0.0, vflip=0.0, rot90=0.0,
                          roi_jitter=0.0, roi_dropout=0.0, roi_distort=0.0,
                          shuffle=False, num_workers=0, fold=fold, batch_size=a.batch_size)
                random.seed(1234); torch.manual_seed(1234)   # 11's seeds
                ld = getattr(module_data, cfg['data_loader']['type'])(**ar)
                m = getattr(module_arch, cfg['arch']['type'])(**cfg['arch']['args']).to(device)
                m.load_state_dict(torch.load(ckpt, map_location=device,
                                             weights_only=False)['state_dict'])
                m.eval()
                got, _ = stitch_iou(m, ld.dataset, list(ld.test_indices), None, device,
                                    bool(cfg['trainer'].get('amp', False)),
                                    getattr(torch, cfg['trainer'].get('amp_dtype','bfloat16')),
                                    a.batch_size)
                worst = max(worst, abs(got - r11.get(fold, got)))
                del m; torch.cuda.empty_cache()
            print(f"  max |diff| vs EXPERIMENT 11: {worst:.2e}  "
                  f"{'OK — stitching reproduces 11' if worst < 5e-3 else '*** harness bug ***'}")

    json.dump({'strengths': S, 'presets': a.presets, 'results': results},
              open(a.out, 'w'), indent=2)
    print(f"\nwrote {a.out}   ({time.time()-t0:.0f}s)")

    print(f"\n{'strength':>9} {'mean IoU':>10} {'sd':>8} {'vs s=0':>9} "
          f"{'ROI IoU':>9} {'ROI px':>8}")
    base_iou = None
    if a.real_variants:
        S = ['real']
    if a.degenerate:
        S = ['full', 'empty', 'wrong']
    for s in S:
        rs = [r for r in results if r['strength'] == s]
        v = [r['slide_iou'] for r in rs]
        if base_iou is None:
            base_iou = float(np.mean(v))
        print(f"{str(s):>9} {np.mean(v):>10.4f} {np.std(v):>8.4f} "
              f"{np.mean(v)-base_iou:>+9.4f} {np.mean([r['roi_iou'] for r in rs]):>9.3f} "
              f"{np.mean([r['roi_surf_px'] for r in rs]):>8.0f}")


if __name__ == '__main__':
    main()
