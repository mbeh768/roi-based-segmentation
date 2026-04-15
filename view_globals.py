import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch


def to_bgr(t, tile_h=256):
    t = t.detach().cpu().float()
    lo, hi = t.min(), t.max()
    t = (t - lo) / (hi - lo + 1e-8)
    arr = (t.numpy() * 255).astype(np.uint8)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        arr = arr.transpose(1, 2, 0)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return cv2.resize(arr, (tile_h, tile_h), interpolation=cv2.INTER_LINEAR)


def label(img, text):
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)
    return img


def main():
    ap = argparse.ArgumentParser(description='Grid of all global.pt in a data/ directory')
    ap.add_argument('data_dir', type=str,
                    help='Path under data/ (e.g. data/ssl or data/[collapsed]64_tiles_per_image)')
    ap.add_argument('--out', type=str, default='globals_grid.png')
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--cols', type=int, default=0,
                    help='Columns (0 = auto square-ish)')
    args = ap.parse_args()

    root = Path(args.data_dir)
    image_dir = root / 'images' if (root / 'images').is_dir() else root
    paths = sorted(image_dir.glob('*/global.pt'))
    assert paths, f'No */global.pt found under {image_dir}'

    tiles = []
    for p in paths:
        t = torch.load(p, weights_only=True)
        tiles.append(label(to_bgr(t, args.tile), p.parent.name))

    n = len(tiles)
    cols = args.cols if args.cols > 0 else int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    blank = np.zeros_like(tiles[0])
    padded = tiles + [blank] * (rows * cols - n)
    grid = np.concatenate(
        [np.concatenate(padded[r * cols:(r + 1) * cols], axis=1) for r in range(rows)],
        axis=0,
    )
    cv2.imwrite(args.out, grid)
    print(f'Saved {n} globals ({rows}x{cols}) to {args.out}')


if __name__ == '__main__':
    main()
