"""Does the segmentation survive a different annotator? (output-stability figure)

The companion to roi_range_figure.py. That one shows how far the ROI *input*
moves under a grid warp; this one runs the model on each warped ROI and shows
what comes out the other side.

Uses a LOCO fold's checkpoint on ITS OWN held-out case, so the slide was never
trained on -- otherwise stability would partly be memorisation.

Reports two different things, which are easy to conflate:
  vs GT        each prediction's IoU against ground truth -- does accuracy hold up
  pairwise     IoU between predictions from different ROIs -- do annotators AGREE
The second is the claim being made. A model could be uniformly mediocre yet
perfectly self-consistent, or accurate on average while flipping between
annotators; only the pair of numbers together says anything.
"""
import argparse, json, random
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader.datasets import _roi_grid_distort

ORIG, PERT = '#2a78d6', '#eb6834'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d8d7d2'


def stitch(model, ds, idxs, roi_override, device, amp, dt, bs=32):
    """Run the fold's test tiles with one fixed ROI, stitch onto the slide canvas."""
    from torch.utils.data.dataloader import default_collate
    canv = None
    with torch.no_grad():
        for s in range(0, len(idxs), bs):
            chunk = idxs[s:s+bs]
            batch = default_collate([ds[i] for i in chunk])
            *raw, target, _g = batch
            raw = [x.to(device) for x in raw]
            raw[3] = roi_override.to(device).unsqueeze(0).expand(len(chunk), -1, -1, -1)
            target = target.to(device)
            with torch.autocast('cuda', dtype=dt, enabled=amp):
                out = model(*raw)
                if isinstance(out, tuple): out = out[0]
            out = out.float()
            if out.shape[-2:] != target.shape[-2:]:
                oh, ow = out.shape[-2:]; th, tw = target.shape[-2:]
                out = out[..., (oh-th)//2:(oh-th)//2+th, (ow-tw)//2:(ow-tw)//2+tw]
            p = torch.sigmoid(out).squeeze(1).cpu().numpy()
            gt = (target > 0.5).float().squeeze(1).cpu().numpy()
            for k, i in enumerate(chunk):
                _, stem, e = ds.items[i]
                if canv is None:
                    H, W = e['image_H'], e['image_W']
                    canv = {'p': np.zeros((H, W), np.float32),
                            'g': np.zeros((H, W), np.float32),
                            'n': np.zeros((H, W), np.uint8)}
                y, x, h, w = e['y'], e['x'], e['height'], e['width']
                canv['p'][y:y+h, x:x+w] += p[k]
                canv['g'][y:y+h, x:x+w] += gt[k]
                canv['n'][y:y+h, x:x+w] += 1
    cov = canv['n'] > 0
    cnt = np.maximum(canv['n'], 1)
    return (canv['p']/cnt > 0.5) & cov, (canv['g']/cnt > 0.5) & cov


def iou(a, b):
    return (np.count_nonzero(a & b) + 1e-9) / (np.count_nonzero(a | b) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-dir', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--name', default='KFOLD_LOCO')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--strength', type=float, default=0.02)
    ap.add_argument('--n-annotators', type=int, default=5)
    ap.add_argument('--out', default='roi_stability.png')
    a = ap.parse_args()

    import data_loader.data_loaders as module_data
    import model.model as module_arch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfgp = ckpt = None
    for c in sorted(Path(a.models_dir).glob('*/config.json')):
        d = json.loads(c.read_text())
        if d.get('name') == a.name and d['data_loader']['args'].get('fold') == a.fold:
            cfgp, ckpt = d, c.parent / 'model_best.pth'; break
    if ckpt is None: raise SystemExit(f"no checkpoint for {a.name} fold {a.fold}")

    args = dict(cfgp['data_loader']['args'])
    args.update(data_dir=a.data_dir, hflip=0.0, vflip=0.0, rot90=0.0, roi_jitter=0.0,
                roi_dropout=0.0, roi_distort=0.0, shuffle=False, num_workers=0,
                fold=a.fold, batch_size=32)
    random.seed(0); torch.manual_seed(0)
    loader = getattr(module_data, cfgp['data_loader']['type'])(**args)
    ds, idxs = loader.dataset, list(loader.test_indices)
    stem = ds.items[idxs[0]][1]

    arch = cfgp['arch']
    model = getattr(module_arch, arch['type'])(**arch['args']).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)['state_dict'])
    model.eval()
    amp = bool(cfgp['trainer'].get('amp', False))
    dt = getattr(torch, cfgp['trainer'].get('amp_dtype', 'bfloat16'))

    base_roi = torch.load(Path(a.data_dir)/'images'/stem/f'roi_{stem}.pt', weights_only=True)
    rois = [base_roi]
    for k in range(a.n_annotators - 1):
        torch.manual_seed(300+k); random.seed(300+k)
        rois.append(_roi_grid_distort(base_roi.clone(), strength=a.strength))

    preds, gt = [], None
    for r in rois:
        p, g = stitch(model, ds, idxs, r, device, amp, dt)
        preds.append(p); gt = g
    vs_gt = [iou(p, gt) for p in preds]
    pair = [iou(preds[i], preds[j]) for i, j in combinations(range(len(preds)), 2)]
    roi_pair = [iou((rois[i][0]>0.5).numpy(), (rois[j][0]>0.5).numpy())
                for i, j in combinations(range(len(rois)), 2)]

    m = gt
    ys, xs = np.where(m); pad = 40
    y0, y1 = max(ys.min()-pad,0), min(ys.max()+pad, m.shape[0])
    x0, x1 = max(xs.min()-pad,0), min(xs.max()+pad, m.shape[1])
    n = len(rois)

    fig, axes = plt.subplots(2, n, figsize=(2.5*n, 5.6))
    for c in range(n):
        r = torch.nn.functional.interpolate(rois[c].unsqueeze(0).float(),
              size=m.shape, mode='nearest')[0,0].numpy()
        ax = axes[0, c]
        ax.imshow(np.zeros_like(m[y0:y1,x0:x1]), cmap='gray', vmin=0, vmax=1)
        ax.contour(r[y0:y1,x0:x1], levels=[0.5], colors=ORIG if c==0 else PERT, linewidths=1.4)
        ax.set_title("annotator 1\n(original)" if c==0 else f"annotator {c+1}\n(simulated)",
                     fontsize=10, color=INK, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(GRID)

        ax = axes[1, c]
        ax.imshow(gt[y0:y1,x0:x1], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.contour(preds[c][y0:y1,x0:x1].astype(float), levels=[0.5],
                   colors=ORIG if c==0 else PERT, linewidths=1.2)
        ax.set_xlabel(f"IoU vs GT {vs_gt[c]:.3f}", fontsize=9, color=INK2, labelpad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(GRID)

    axes[0,0].set_ylabel("ROI given to the model", fontsize=10, color=INK2, labelpad=8)
    axes[1,0].set_ylabel("resulting segmentation\n(grey = ground truth)", fontsize=10,
                         color=INK2, labelpad=8)
    fig.suptitle(f"Segmentation stability across annotators — {stem} (held out, fold {a.fold})",
                 fontsize=13, color=INK, y=0.99)
    fig.text(0.5, 0.925,
             f"ROIs differ by IoU {np.mean(roi_pair):.3f} on average, yet the predictions agree "
             f"at IoU {np.mean(pair):.3f}   ·   accuracy vs GT {np.mean(vs_gt):.3f} ± {np.std(vs_gt):.3f}",
             ha='center', fontsize=10, color=INK2)
    fig.tight_layout(rect=[0,0,1,0.90])
    fig.savefig(a.out, dpi=140, facecolor='#fcfcfb')
    print(f"wrote {a.out}\n")
    print(f"slide {stem}, fold {a.fold}, strength {a.strength}, {n} annotators")
    print("  ROI pairwise IoU (inputs)      %.4f +/- %.4f" % (np.mean(roi_pair), np.std(roi_pair)))
    print("  prediction pairwise IoU        %.4f +/- %.4f" % (np.mean(pair), np.std(pair)))
    print("  prediction vs GT               %.4f +/- %.4f  (range %.4f-%.4f)"
          % (np.mean(vs_gt), np.std(vs_gt), min(vs_gt), max(vs_gt)))
    json.dump({'stem':stem,'fold':a.fold,'strength':a.strength,
               'roi_pairwise':roi_pair,'pred_pairwise':pair,'vs_gt':vs_gt},
              open(Path(a.out).with_suffix('.json'),'w'), indent=2)


if __name__ == '__main__':
    main()
