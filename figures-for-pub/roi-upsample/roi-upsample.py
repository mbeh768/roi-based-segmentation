"""How much of the slide-level score is the ROI, and how much is the network?

Three quantities per slide, all at NATIVE TIFF resolution over the whole image,
scored by slide_level_iou.py's own score_native so they are directly comparable
to the reported model metric:

  ceiling   masks/<stem>/global.pt upsampled to native. The downsampled GT is a
            PERFECT 256-grid ROI, so this is the best any 256-grid ROI could
            possibly score -- it isolates what the 32x bottleneck costs before
            annotation error enters at all.
  roi       roi_<stem>.pt upsampled to native. The drawn ROI alone, with no
            network: the floor a model given that ROI must beat to be doing
            real work.
  model     read from the fold sweep's <name>_slide_iou.json.

ceiling -> roi is annotation error; roi -> model is the network's contribution.
Reporting roi without ceiling would charge resolution loss to the annotator.

Coverage is the union of the tile rectangles that survived roi_exclusion, built
from tile coordinates alone -- tile CONTENTS never enter it, so no tensor is
opened and no model is loaded. CPU only.

Usage:
    python figures-for-pub/roi-upsample/roi-upsample.py
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'train.py').exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'utils'))

import data_loader.data_loaders as module_data     # noqa: E402
from slide_level_iou import score_native           # noqa: E402


def to_canvas(t, shape):
    """Put a (1, 256, 256) ROI on the tiling canvas, which is what score_native
    expects as a prediction -- it does the canvas -> native step itself."""
    m = (t.squeeze(0).numpy() > 0.5).astype(np.uint8)
    return cv2.resize(m, shape[::-1], interpolation=cv2.INTER_NEAREST) > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', default=str(ROOT / 'config.json'))
    ap.add_argument('--model-json',
                    default=str(ROOT / 'figures-for-pub' / 'baseline'
                                / 'k-fold-baseline-2026-09-02_slide_iou.json'))
    ap.add_argument('--out', default=str(HERE / 'roi-upsample.json'))
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text())
    raw_dir = cfg['preprocess']['raw_image_directory']
    data_dir = ROOT / cfg['data_loader']['args']['data_dir']
    args = {k: v for k, v in cfg['data_loader']['args'].items()
            if k not in ('hflip', 'vflip', 'rot90', 'shuffle', 'num_workers', 'batch_size')}
    args.update(shuffle=False, num_workers=0, batch_size=1)
    ds = module_data.LocalGlobalDataLoader(**args).dataset

    covered = {}
    for _, stem, e in ds.items:
        if stem not in covered:
            covered[stem] = np.zeros((e['image_H'], e['image_W']), bool)
        covered[stem][e['y']:e['y'] + e['height'], e['x']:e['x'] + e['width']] = True

    model = {}
    for f in json.loads(Path(a.model_json).read_text())['folds']:
        model.update({k: (v['iou'], v['dice']) for k, v in f['per_slide'].items()})

    rows = []
    print(f"{'slide':<22}{'ceiling':>9}{'roi':>9}{'model':>9}   {'annot':>8}{'network':>9}")
    for stem in sorted(covered):
        cov = covered[stem]
        gt256 = torch.load(data_dir / 'masks' / stem / 'global.pt', weights_only=True)
        roi = torch.load(data_dir / 'images' / stem / f'roi_{stem}.pt', weights_only=True)

        *_, ceil_rec = score_native(to_canvas(gt256, cov.shape), cov, stem, raw_dir)
        *_, roi_rec = score_native(to_canvas(roi, cov.shape), cov, stem, raw_dir)
        m_iou, m_dice = model.get(stem, (float('nan'), float('nan')))
        rows.append({'stem': stem,
                     'ceiling': ceil_rec['iou'], 'ceiling_dice': ceil_rec['dice'],
                     'roi': roi_rec['iou'], 'roi_dice': roi_rec['dice'],
                     'model': m_iou, 'model_dice': m_dice,
                     'native_hw': ceil_rec['native_hw'], 'gt_px': ceil_rec['gt_px']})
        print(f"{stem:<22}{ceil_rec['iou']:>9.4f}{roi_rec['iou']:>9.4f}{m_iou:>9.4f}   "
              f"{ceil_rec['iou'] - roi_rec['iou']:>8.4f}{m_iou - roi_rec['iou']:>9.4f}")

    Path(a.out).write_text(json.dumps({'slides': rows}, indent=2))
    col = lambda k: np.array([r[k] for r in rows], float)
    print(f"\nmean  ceiling {col('ceiling').mean():.4f}   roi {col('roi').mean():.4f}   "
          f"model {col('model').mean():.4f}")
    print(f"      resolution cost {1 - col('ceiling').mean():.4f}   "
          f"annotation error {(col('ceiling') - col('roi')).mean():.4f}   "
          f"network gain {(col('model') - col('roi')).mean():.4f}")
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
