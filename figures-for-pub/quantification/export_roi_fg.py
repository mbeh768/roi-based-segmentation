"""Materialise the ROI-minus-foreground arm as masks, with per-slide IoU.

quantify_density.py builds this arm inline, so it exists nowhere on disk, and
`masks/stub-roi` is the plain upsampled ROI, not this. Every step is imported from
there rather than restated. No published per-slide IoU exists, so it is computed
here against the source mask at native resolution.

    python figures-for-pub/quantification/export_roi_fg.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'train.py').exists())
sys.path.insert(0, str(HERE))

from quantify_density import (arm_mask, channel_mean, close_mask,  # noqa: E402
                              drop_islands, find, load_mask, load_rgb,
                              otsu_foreground_mask)

ARM = 'ROI - foreground'


class Ctx:
    """The attribute surface ARMS[ARM] reaches for: nothing more."""
    def __init__(self, data_dir, raw_dir):
        self.data_dir, self.raw_dir = data_dir, raw_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cfg = json.loads((ROOT / 'config.json').read_text())
    ap.add_argument('--data-dir', default=str(ROOT / cfg['data_loader']['args']['data_dir']))
    ap.add_argument('--raw-dir', default=cfg['preprocess']['raw_image_directory'])
    ap.add_argument('--outdir', default=str(HERE / 'masks' / 'roi-fg'))
    ap.add_argument('--scores', default=str(HERE / 'roi_fg_scores.json'))
    ap.add_argument('--slides', nargs='*', default=None)
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    stems = a.slides or sorted(
        p.stem for p in (Path(a.data_dir) / 'images').iterdir() if p.is_dir())
    ctx_args = Ctx(a.data_dir, a.raw_dir)

    rows = []
    for stem in stems:
        R, G, B = load_rgb(find(a.raw_dir, stem, 'image'))
        gt = load_mask(find(a.raw_dir, stem, 'mask'))
        if gt.shape != R.shape:
            raise SystemExit(f'{stem}: mask {gt.shape} != image {R.shape}')

        scanned = (R > 0) | (G > 0) | (B > 0)
        foreground, _, dropped = drop_islands(
            close_mask(otsu_foreground_mask(channel_mean(R, G, B), scanned)))
        del scanned
        ctx = {'shape': R.shape, 'foreground': foreground}
        del R, G, B

        mask = arm_mask(ARM, ctx_args, stem, ctx)
        inter, union = int((mask & gt).sum()), int((mask | gt).sum())
        iou = inter / max(union, 1)
        dice = 2 * inter / max(int(mask.sum()) + int(gt.sum()), 1)
        np.savez_compressed(out / f'{stem}.npz', pred=mask, stem=stem,
                            native_hw=np.array(mask.shape))
        rows.append({'stem': stem, 'arm': 'roi-fg', 'iou': iou, 'dice': dice,
                     'pred_px': int(mask.sum()), 'gt_px': int(gt.sum()),
                     'foreground_frac': float(foreground.mean()),
                     'native_hw': list(mask.shape)})
        print(f'  {stem:22} foreground {foreground.mean()*100:5.1f}%  -{dropped} islands'
              f'   iou {iou:.4f}  dice {dice:.4f}', flush=True)
        del mask, gt, foreground, ctx

    Path(a.scores).write_text(json.dumps({'roi-fg': {'slides': rows}}, indent=1))
    v = np.array([r['iou'] for r in rows])
    print(f'\nwrote {len(rows)} masks to {out}\nwrote {a.scores}')
    print(f'  IoU mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}  '
          f'range {v.min():.4f}..{v.max():.4f}')


if __name__ == '__main__':
    main()
