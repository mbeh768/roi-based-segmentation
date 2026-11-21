"""Render each slide's SOURCE mask onto the canvas grid, once, and cache it.

The `gt` inside canvases/<stem>.npz is `(accumulated/count > 0.5) & covered`, so it
is limited to the pixels a tile reached -- 7-51% of the section. That is right for
the network, which only predicts there, but SAM, microSAM and ROI-FG are whole-image
methods: scored against a covered-limited GT, everything they output elsewhere would
read as a false positive.

So the agreement figures take their GT from here instead. The mask is read at native
resolution and area-downsampled to the canvas shape, thresholded at 0.5 -- the same
majority rule preprocess.py applies when it builds the tiles.

    python figures-for-pub/qualitative/export_native_gt.py
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'train.py').exists())


def native_mask(raw_dir, stem):
    """Same source and threshold as slide_level_metrics.score_native."""
    hits = sorted(Path(raw_dir).glob(f'*/{stem}_mask.*'))
    if not hits:
        raise SystemExit(f'no source mask for {stem} under {raw_dir}')
    gt = tifffile.TiffFile(hits[0]).series[0].asarray()
    if gt.ndim == 3:
        gt = gt[..., 0] if gt.shape[-1] <= 4 else gt[0]
    return (gt > 127), hits[0].name


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--canvases', default=str(HERE / 'canvases'),
                    help='sets the target shape per slide')
    ap.add_argument('--outdir', default=str(HERE / 'gt-native'))
    ap.add_argument('--raw-dir',
                    default=json.loads((ROOT / 'config.json').read_text())
                    ['preprocess']['raw_image_directory'])
    ap.add_argument('--slides', nargs='*', default=None)
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    stems = a.slides or sorted(p.stem for p in Path(a.canvases).glob('*.npz'))

    for stem in stems:
        shape = np.load(Path(a.canvases) / f'{stem}.npz')['gt'].shape
        gt, name = native_mask(a.raw_dir, stem)
        # uint8 rather than float: a 1.4 Gpx mask as float32 is 5.7 GB
        small = cv2.resize(gt.astype(np.uint8) * 255, (shape[1], shape[0]),
                           interpolation=cv2.INTER_AREA) > 127
        np.savez_compressed(out / f'{stem}.npz', gt=small,
                            native_hw=np.array(gt.shape), stem=stem)
        print(f'  {stem:22} {name:34} {gt.shape[1]}x{gt.shape[0]} -> '
              f'{shape[1]}x{shape[0]}   {100 * small.mean():5.2f}% lumen')
        del gt, small

    print(f'wrote {len(stems)} masks to {out}')


if __name__ == '__main__':
    main()
