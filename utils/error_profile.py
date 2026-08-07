"""Characterise WHERE the held-out error actually is, per slide.

Decides whether a continuity/topology loss is worth trying at all. Rebuilds the
stitched prediction and ground truth for each fold's test case (same procedure
as utils/slide_level_iou.py) and decomposes the residual error into:

  islands       false-positive components not touching any true component, and
                how much of the total FP mass they hold. This is the "specks"
                failure the 9F/9G/9H pocket loss targeted -- if the mass is
                small, a continuity loss cannot buy much no matter how it is
                formulated.
  holes         false negatives fully enclosed by predicted foreground.
  boundary      error within +/-8 px of the true boundary, i.e. ordinary edge
                imprecision, which no topology term fixes.
  bulk          everything else: large connected mis-segmentations.

A high boundary share means the ceiling is annotation/resolution, not topology.
"""
import argparse, json, random, sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from utils import read_json, write_json

BOUNDARY_PX = 8


def crop_to_target(o, t):
    if o.shape[-2:] == t.shape[-2:]:
        return o
    oh, ow = o.shape[-2:]; th, tw = t.shape[-2:]
    return o[..., (oh - th) // 2:(oh - th) // 2 + th, (ow - tw) // 2:(ow - tw) // 2 + tw]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--models-dir', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', default='error_profile.json')
    ap.add_argument('--batch-size', type=int, default=32)
    a = ap.parse_args()

    import data_loader.data_loaders as module_data
    import model.model as module_arch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    runs = {}
    for cp in sorted(Path(a.models_dir).glob('*/config.json')):
        c = read_json(cp)
        if c.get('name') != a.name:
            continue
        ar = c['data_loader']['args']
        if ar.get('n_folds') and (cp.parent / 'model_best.pth').exists():
            runs[ar.get('fold', 0)] = (c, cp.parent / 'model_best.pth')

    out = []
    for fold in sorted(runs):
        cfg, ckpt = runs[fold]
        args = dict(cfg['data_loader']['args'])
        args['data_dir'] = a.data_dir
        args.update(hflip=0.0, vflip=0.0, rot90=0.0, roi_jitter=0.0, roi_dropout=0.0,
                    roi_distort=0.0, shuffle=False, num_workers=0, fold=fold,
                    batch_size=a.batch_size)
        random.seed(1234); torch.manual_seed(1234)
        loader = getattr(module_data, cfg['data_loader']['type'])(**args)
        ds, test_idx = loader.dataset, list(loader.test_indices)

        arch = cfg['arch']
        model = getattr(module_arch, arch['type'])(**arch['args']).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=False)['state_dict'])
        model.eval()
        amp = bool(cfg['trainer'].get('amp', False))
        dt = getattr(torch, cfg['trainer'].get('amp_dtype', 'bfloat16'))

        canv = {}
        for i in test_idx:
            _, stem, e = ds.items[i]
            if stem not in canv:
                canv[stem] = {'p': np.zeros((e['image_H'], e['image_W']), np.float32),
                              'g': np.zeros((e['image_H'], e['image_W']), np.float32),
                              'n': np.zeros((e['image_H'], e['image_W']), np.uint8)}
        with torch.no_grad():
            for s in range(0, len(test_idx), a.batch_size):
                ch = test_idx[s:s + a.batch_size]
                b = torch.utils.data.dataloader.default_collate([ds[i] for i in ch])
                *raw, target, _g = b
                inp = [x.to(device) for x in raw]
                target = target.to(device)
                with torch.autocast('cuda', dtype=dt, enabled=amp):
                    o = model(*inp)
                    if isinstance(o, tuple):
                        o = o[0]
                o = crop_to_target(o.float(), target)
                pn = torch.sigmoid(o).squeeze(1).cpu().numpy()
                gn = (target > 0.5).float().squeeze(1).cpu().numpy()
                for k, idx in enumerate(ch):
                    _, stem, e = ds.items[idx]
                    y, x, h, w = e['y'], e['x'], e['height'], e['width']
                    c = canv[stem]
                    c['p'][y:y + h, x:x + w] += pn[k]
                    c['g'][y:y + h, x:x + w] += gn[k]
                    c['n'][y:y + h, x:x + w] += 1

        for stem, c in sorted(canv.items()):
            cov = c['n'] > 0
            cnt = np.maximum(c['n'], 1)
            pred = (c['p'] / cnt > 0.5) & cov
            gt = (c['g'] / cnt > 0.5) & cov
            fp = pred & ~gt
            fn = gt & ~pred
            inter = np.count_nonzero(pred & gt)
            union = np.count_nonzero(pred | gt)

            # false-positive components that touch no true foreground -> islands
            lab, n = ndimage.label(pred)
            island_px = 0
            n_islands = 0
            if n:
                touches = set(np.unique(lab[gt]))
                touches.discard(0)
                for cid, sz in zip(*np.unique(lab[lab > 0], return_counts=True)):
                    if cid not in touches:
                        island_px += int(sz); n_islands += 1

            # false negatives enclosed by prediction -> holes
            filled = ndimage.binary_fill_holes(pred)
            hole_px = int(np.count_nonzero(filled & ~pred & gt))

            # error near the true boundary -> ordinary edge imprecision
            if gt.any() and not gt.all():
                # distance to the GT boundary, measured from whichever side the
                # pixel is on. Each transform is 0 on the opposite side, so they
                # must be selected between, not OR-ed -- OR-ing marks every pixel.
                dist = np.where(gt,
                                ndimage.distance_transform_edt(gt),
                                ndimage.distance_transform_edt(~gt))
                band = dist <= BOUNDARY_PX
            else:
                band = np.zeros_like(gt)
            err = fp | fn
            err_px = int(np.count_nonzero(err))
            boundary_px = int(np.count_nonzero(err & band))

            out.append({
                'fold': fold, 'stem': stem,
                'iou': (inter + 1e-6) / (union + 1e-6),
                'gt_px': int(np.count_nonzero(gt)), 'pred_px': int(np.count_nonzero(pred)),
                'fp_px': int(np.count_nonzero(fp)), 'fn_px': int(np.count_nonzero(fn)),
                'err_px': err_px,
                'island_px': island_px, 'n_islands': n_islands,
                'hole_px': hole_px,
                'boundary_err_px': boundary_px,
                'frac_err_boundary': boundary_px / max(err_px, 1),
                'frac_err_islands': island_px / max(err_px, 1),
                'frac_err_holes': hole_px / max(err_px, 1),
                'frac_fp_islands': island_px / max(int(np.count_nonzero(fp)), 1),
            })
            print(f"  fold {fold:>2} {stem:<22} iou={out[-1]['iou']:.4f} "
                  f"err={err_px:>9,}  boundary={out[-1]['frac_err_boundary']*100:5.1f}%  "
                  f"islands={out[-1]['frac_err_islands']*100:5.1f}% ({n_islands})  "
                  f"holes={out[-1]['frac_err_holes']*100:4.1f}%", flush=True)
        del model, canv
        torch.cuda.empty_cache()

    tot_err = sum(r['err_px'] for r in out)
    agg = {
        'slides': out,
        'total_err_px': tot_err,
        'frac_err_boundary': sum(r['boundary_err_px'] for r in out) / max(tot_err, 1),
        'frac_err_islands': sum(r['island_px'] for r in out) / max(tot_err, 1),
        'frac_err_holes': sum(r['hole_px'] for r in out) / max(tot_err, 1),
        'total_fp_px': sum(r['fp_px'] for r in out),
        'total_fn_px': sum(r['fn_px'] for r in out),
        'boundary_px': BOUNDARY_PX,
    }
    agg['frac_err_bulk'] = max(0.0, 1 - agg['frac_err_boundary']
                               - agg['frac_err_islands'] - agg['frac_err_holes'])
    write_json(agg, a.out)
    print(f"\n=== pooled over {len(out)} slides ({tot_err:,} error px) ===")
    print(f"  near boundary (+/-{BOUNDARY_PX}px) : {agg['frac_err_boundary']*100:5.1f}%")
    print(f"  FP islands (no GT contact)  : {agg['frac_err_islands']*100:5.1f}%")
    print(f"  enclosed holes              : {agg['frac_err_holes']*100:5.1f}%")
    print(f"  bulk / other                : {agg['frac_err_bulk']*100:5.1f}%")
    print(f"  FP:FN = {agg['total_fp_px']:,} : {agg['total_fn_px']:,}")
    print(f"\nwrote {a.out}")


if __name__ == '__main__':
    main()
