"""
Run LocalGlobalUNet inference on a raw TIFF.
The global image and all local tiles are preprocessed on the fly.
Predictions are stitched back into a full-image view with tile boundary lines.

Preprocess params (tile_size, stride, tiling_size, global_downsample) are read
from -c config.json.

Usage:
    # single image
    python inference_gl.py --image slide.tif -m <ckpt> -c config.json
    python inference_gl.py --image slide.tif --mask mask.tif -m <ckpt> -c config.json

    # whole dataset (close window to advance)
    python inference_gl.py --dataset-pairs -m <ckpt> -c config.json
"""
import argparse
import json
import random
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from preprocess import Preprocessor
from inference_global import load_model, tif_to_tensor, to_display
from local_inference import tif_to_tiles, stitch


def run_one(img_path, mask_path, model, device, p, threshold, title, output=None):
    print(f"\nPreprocessing {img_path.name}...")
    tiles, ys, xs, H, W, tile_size, stride, full_arr = tif_to_tiles(img_path, p, is_mask=False)
    glob = tif_to_tensor(img_path, p.c, is_mask=False)   # (3, global_H, global_W)

    mask_stitched = None
    if mask_path is not None:
        print(f"Preprocessing mask {mask_path.name}...")
        m_tiles, *m_rest = tif_to_tiles(mask_path, p, is_mask=True)
        m_ys, m_xs, m_H, m_W, m_ts, m_st = m_rest[:6]
        mask_stitched = stitch([t.squeeze() for t in m_tiles], m_ys, m_xs, m_H, m_W, m_ts, m_st)

    # batch all tiles; broadcast the single global image across the batch
    N = len(tiles)
    local_batch = torch.stack(tiles).to(device)                    # (N, 3, tile_H, tile_W)
    glob_batch  = glob.unsqueeze(0).expand(N, -1, -1, -1).to(device)  # (N, 3, gH, gW)

    pos_list = [
        torch.tensor([y / H, x / W, (y + tile_size) / H, (x + tile_size) / W], dtype=torch.float32)
        for y in ys for x in xs
    ]
    pos_batch = torch.stack(pos_list).to(device)                   # (N, 4)

    with torch.no_grad():
        logits = model(local_batch, glob_batch, pos_batch)         # (N, 1, tile_H, tile_W)
    probs_tiles = torch.sigmoid(logits).squeeze(1).cpu()           # (N, tile_H, tile_W)

    prob_map = stitch(list(probs_tiles), ys, xs, H, W, tile_size, stride)
    pred_map = (prob_map > threshold).float()

    # ── figure ───────────────────────────────────────────────────────────────
    import numpy as np
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
                        help='config JSON with preprocess section')
    parser.add_argument('--mask',         default=None,  help='ground-truth mask TIFF (single-image mode)')
    parser.add_argument('--output',       default=None,  help='save figure here instead of displaying')
    parser.add_argument('--threshold',    default=0.5, type=float,
                        help='sigmoid threshold for binary prediction (default 0.5)')
    main(parser.parse_args())
