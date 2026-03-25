"""
Run local (tile-based) inference on a raw TIFF using a trained SimpleUNet.
Splits the TIFF into tiles using the same pipeline as preprocess.py, runs the
model on all tiles, then stitches predictions back into a full-image view.

Preprocess params (tile_size, stride, tiling_size) are read from -c config.json.

Usage:
    # single image
    python local_inference.py --image slide.tif -m <ckpt> -c config.json
    python local_inference.py --image slide.tif --mask mask.tif -m <ckpt> -c config.json

    # whole dataset (close window to advance)
    python local_inference.py --dataset-pairs -m <ckpt> -c config.json
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

import model.model as module_arch
from preprocess import Preprocessor
from inference_global import load_model, to_display


def tif_to_tiles(path, p, is_mask=False):
    """
    Load TIFF at tiling_size resolution and cut into tile_size patches.
    Returns (tiles, ys, xs, full_arr) where tiles is a list of (C,H,W) tensors
    and ys/xs are the grid coordinates used for stitching.
    """
    tile_size   = p.c['preprocess']['tile_size']
    stride      = p.c['preprocess']['stride']
    tiling_size = p.c['preprocess']['tiling_size']

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    arr = p.load_and_format(path, interp, standardize=not is_mask,
                            is_mask=is_mask, target_size=tiling_size)  # (C, H, W)

    img_hwc = np.transpose(arr, (1, 2, 0))
    H, W = img_hwc.shape[:2]

    ys = list(range(0, H - tile_size + 1, stride))
    xs = list(range(0, W - tile_size + 1, stride))

    tiles = []
    for y in ys:
        for x in xs:
            patch = img_hwc[y:y+tile_size, x:x+tile_size]
            tiles.append(torch.from_numpy(np.transpose(patch, (2, 0, 1)).copy()))

    return tiles, ys, xs, H, W, tile_size, stride, arr


def stitch(tile_maps, ys, xs, H, W, tile_size, stride):
    """Average-stitch single-channel (H,W) tile predictions into a full canvas."""
    canvas = torch.zeros(H, W)
    count  = torch.zeros(H, W)
    for i, (y, x) in enumerate([(y, x) for y in ys for x in xs]):
        canvas[y:y+tile_size, x:x+tile_size] += tile_maps[i]
        count[y:y+tile_size,  x:x+tile_size] += 1
    return canvas / count.clamp(min=1)


def run_one(img_path, mask_path, model, device, p, threshold, title, output=None):
    print(f"\nPreprocessing {img_path.name}...")
    tiles, ys, xs, H, W, tile_size, stride, full_arr = tif_to_tiles(img_path, p, is_mask=False)

    mask_stitched = None
    if mask_path is not None:
        print(f"Preprocessing mask {mask_path.name}...")
        m_tiles, *m_rest = tif_to_tiles(mask_path, p, is_mask=True)
        m_ys, m_xs, m_H, m_W, m_ts, m_st = m_rest[:6]
        mask_stitched = stitch([t.squeeze() for t in m_tiles], m_ys, m_xs, m_H, m_W, m_ts, m_st)

    # batch all tiles through the model at once
    batch = torch.stack(tiles).to(device)             # (N, C, tile_H, tile_W)
    with torch.no_grad():
        logits = model(batch)                         # (N, 1, tile_H, tile_W)
    probs_tiles = torch.sigmoid(logits).squeeze(1).cpu()  # (N, tile_H, tile_W)

    prob_map = stitch(list(probs_tiles), ys, xs, H, W, tile_size, stride)
    pred_map = (prob_map > threshold).float()

    # ── figure ───────────────────────────────────────────────────────────────
    n_cols = 4 if mask_stitched is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    fig.suptitle(title)

    axes[0].imshow(to_display(torch.from_numpy(full_arr)))
    axes[0].set_title('Input')

    axes[1].imshow(prob_map.numpy(), cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Probability')

    axes[2].imshow(pred_map.numpy(), cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'Prediction (t={threshold})')

    if mask_stitched is not None:
        axes[3].imshow(mask_stitched.numpy(), cmap='gray', vmin=0, vmax=1)
        axes[3].set_title('Ground truth')

    for ax in axes:
        ax.axis('off')
        for y in ys[1:]:
            ax.axhline(y, color='cyan', linewidth=0.5, alpha=0.6)
        for x in xs[1:]:
            ax.axvline(x, color='cyan', linewidth=0.5, alpha=0.6)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved → {output}")
        plt.close(fig)
    else:
        plt.show()


def main(args):
    model, device, _ = load_model(args.model)

    with open(args.config) as f:
        raw_config = json.load(f)
    p = Preprocessor(raw_config)

    if args.dataset_pairs:
        image_paths, mask_paths = p.find_source_filepaths()
        pairs = list(zip(image_paths, mask_paths))
        random.shuffle(pairs)
        print(f"Found {len(pairs)} image/mask pairs")
        for i, (img_path, mask_path) in enumerate(pairs):
            title = f"{img_path.name}  [{i+1}/{len(pairs)}]"
            run_one(img_path, mask_path, model, device, p, args.threshold, title, args.output)
    else:
        img_path  = Path(args.image)
        mask_path = Path(args.mask) if args.mask else None
        run_one(img_path, mask_path, model, device, p, args.threshold, img_path.name, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--image',         help='path to a single input TIFF')
    src.add_argument('--dataset-pairs', action='store_true',
                     help='loop through all pairs from config raw_image_directory')

    parser.add_argument('-m', '--model',  required=True, help='path to checkpoint (.pth)')
    parser.add_argument('-c', '--config', required=True,
                        help='config JSON with preprocess section (tile_size, stride, tiling_size)')
    parser.add_argument('--mask',         default=None,  help='ground-truth mask TIFF (single-image mode)')
    parser.add_argument('--output',       default=None,  help='save figure here instead of displaying')
    parser.add_argument('--threshold',    default=0.5, type=float,
                        help='sigmoid threshold for binary prediction (default 0.5)')
    main(parser.parse_args())
