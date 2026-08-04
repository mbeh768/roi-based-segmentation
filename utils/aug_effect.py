"""Isolate the test-time augmentation effect.

Everything is held fixed except the augmentation flags: same batch size, same
seed (so the ROI variant choice matches), same tile order. Reports POOLED IoU
(global intersection/union over all test tiles), which -- unlike the per-batch
mean the sweep logged -- does not depend on how tiles are grouped into batches.
"""
import json, random, sys
from pathlib import Path

sys.path.insert(0, '/mnt/1TBssd/mayo_kfold/att_vent_seg')
import numpy as np
import torch
import data_loader.data_loaders as module_data
import model.model as module_arch

MODELS = Path('/mnt/1TBssd/mayo_kfold/saved/models')
DATA = '/mnt/1TBssd/mayo_kfold/data/[collapsed]t8192_ctx512_inner512_old'
device = torch.device('cuda')


def score(cfg, ckpt, fold, augment, bs=40, seed=1234):
    args = dict(cfg['data_loader']['args'])
    args['data_dir'] = DATA
    if augment:
        args.update(hflip=0.5, vflip=0.5, rot90=0.75)     # exactly as the sweep ran
    else:
        args.update(hflip=0.0, vflip=0.0, rot90=0.0)
    args.update(shuffle=False, num_workers=0, fold=fold, batch_size=bs)
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)

    loader = getattr(module_data, cfg['data_loader']['type'])(**args)
    ds, test_idx = loader.dataset, list(loader.test_indices)

    arch = cfg['arch']
    model = getattr(module_arch, arch['type'])(**arch['args']).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=False)['state_dict'])
    model.eval()
    amp = bool(cfg['trainer'].get('amp', False))
    dt = getattr(torch, cfg['trainer'].get('amp_dtype', 'bfloat16'))

    I = U = 0.0
    batch_ious = []
    with torch.no_grad():
        for s in range(0, len(test_idx), bs):
            chunk = test_idx[s:s + bs]
            b = torch.utils.data.dataloader.default_collate([ds[i] for i in chunk])
            *raw, target, _g = b
            inp = [x.to(device) for x in raw]
            target = target.to(device)
            with torch.autocast('cuda', dtype=dt, enabled=amp):
                o = model(*inp)
                if isinstance(o, tuple):
                    o = o[0]
            o = o.float()
            if o.shape[-2:] != target.shape[-2:]:
                oh, ow = o.shape[-2:]; th, tw = target.shape[-2:]
                o = o[..., (oh-th)//2:(oh-th)//2+th, (ow-tw)//2:(ow-tw)//2+tw]
            p = (torch.sigmoid(o) > 0.5).float()
            t = (target > 0.5).float()
            i_ = (p * t).sum(); u_ = p.sum() + t.sum() - i_
            I += float(i_); U += float(u_)
            batch_ious.append(float((i_ + 1e-6) / (u_ + 1e-6)))
    del model; torch.cuda.empty_cache()
    return (I + 1e-6) / (U + 1e-6), float(np.mean(batch_ious))


runs = {}
for cp in sorted(MODELS.glob('*/config.json')):
    c = json.loads(cp.read_text())
    if c.get('name') != 'KFOLD_LOCO':
        continue
    a = c['data_loader']['args']
    if a.get('n_folds') and (cp.parent / 'model_best.pth').exists():
        runs[a.get('fold', 0)] = (c, cp.parent / 'model_best.pth')

print(f"{'fold':>4} {'case':<24} {'pooled_AUG':>11} {'pooled_NOAUG':>13} {'delta':>8}")
deltas, bm_deltas = [], []
for fold in sorted(runs):
    cfg, ckpt = runs[fold]
    case = ', '.join(cfg['data_loader']['args'].get('test_cases', [])) or ''
    on_p, on_b = score(cfg, ckpt, fold, augment=True)
    off_p, off_b = score(cfg, ckpt, fold, augment=False)
    d = off_p - on_p
    deltas.append(d); bm_deltas.append(off_b - on_b)
    print(f"{fold:>4} {case:<24} {on_p:>11.4f} {off_p:>13.4f} {d:>+8.4f}", flush=True)

print(f"\nPOOLED IoU, augmentation off minus on:")
print(f"  mean {np.mean(deltas):+.4f}   sd {np.std(deltas, ddof=1):.4f}   "
      f"max |delta| {max(abs(x) for x in deltas):.4f}")
print(f"batch-mean IoU delta (same batching): mean {np.mean(bm_deltas):+.4f}")
