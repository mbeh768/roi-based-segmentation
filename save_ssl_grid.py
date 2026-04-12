import argparse
import json
from pathlib import Path

import numpy as np
import torch
import cv2

import model.model as module_arch
import data_loader.data_loaders as module_data


def to_img(t):
    t = t.detach().cpu().float()
    lo, hi = t.min(), t.max()
    t = (t - lo) / (hi - lo + 1e-8)
    arr = (t.numpy() * 255).astype(np.uint8)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        arr = arr.transpose(1, 2, 0)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def label(img, text):
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('-n', '--n_samples', type=int, default=12)
    ap.add_argument('-o', '--out', default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    ckpt_path = Path(args.resume)
    run_dir = ckpt_path.parent
    with open(run_dir / 'config.json') as f:
        cfg = json.load(f)

    arch_args = cfg['arch']['args']
    model = getattr(module_arch, cfg['arch']['type'])(**arch_args).to(args.device)
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    loader = getattr(module_data, cfg['data_loader']['type'])(**cfg['data_loader']['args'])
    val_loader = loader.split_validation()

    out_dir = Path(args.out) if args.out else run_dir / 'recon_grid'
    out_dir.mkdir(parents=True, exist_ok=True)

    collected = 0
    rows = []
    with torch.no_grad():
        for batch in val_loader:
            aug_global, pos, clean_global, local_target = [b.to(args.device) for b in batch]
            global_recon, local_recon, _ = model(aug_global, pos, local_target)

            B = aug_global.shape[0]
            for i in range(B):
                if collected >= args.n_samples:
                    break
                gH = 256
                ag = cv2.resize(to_img(aug_global[i]), (gH, gH))
                gr = cv2.resize(to_img(global_recon[i]), (gH, gH))
                cg = cv2.resize(to_img(clean_global[i]), (gH, gH))
                lt = cv2.resize(to_img(local_target[i]), (gH, gH))
                lr = cv2.resize(to_img(local_recon[i]), (gH, gH))

                panels = [
                    label(ag.copy(), 'aug_in'),
                    label(gr.copy(), 'global_recon'),
                    label(cg.copy(), 'clean_target'),
                    label(lt.copy(), 'local_target'),
                    label(lr.copy(), 'local_recon'),
                ]
                row = np.concatenate(panels, axis=1)
                rows.append(row)
                collected += 1
            if collected >= args.n_samples:
                break

    grid = np.concatenate(rows, axis=0)
    out_path = out_dir / f'grid_{ckpt_path.stem}.png'
    cv2.imwrite(str(out_path), grid)
    print(f'Saved {collected} samples to {out_path}')


if __name__ == '__main__':
    main()
