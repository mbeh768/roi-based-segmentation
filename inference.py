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

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

import model.model as module_arch
from preprocess import Preprocessor


def load_model(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = config.init_obj('arch', module_arch)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    print(f"Loaded {ckpt['arch']} from epoch {ckpt['epoch']}")
    return model, device, config


def tif_to_tensor(path, config, is_mask=False):
    """Load a TIFF and return a (C, H, W) float32 tensor via the preprocess pipeline."""
    p = Preprocessor(config)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    imagenet_norm = config['preprocess'].get('imagenet_norm', False) if not is_mask else False
    arr = p.load_and_format(path, interp, standardize=not is_mask,
                            is_mask=is_mask, imagenet_norm=imagenet_norm)
    if not is_mask and config['preprocess'].get('collapse', False):
        arr = p.collapse_channels(arr)
    return torch.from_numpy(arr)


def to_display(tensor):
    """(C, H, W) float32 → (H, W, C) uint8. Matches Preprocessor._make_vis."""
    return Preprocessor(None)._make_vis(tensor.cpu().float().numpy())


def tif_to_tiles(path, p, is_mask=False):
    """Load TIFF at tiling_size resolution and cut into tile_size patches."""
    tile_size   = p.c['preprocess']['tile_size']
    stride      = p.c['preprocess']['stride']
    tiling_size = p.c['preprocess']['tiling_size']

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    imagenet_norm = p.c['preprocess'].get('imagenet_norm', False)
    arr = p.load_and_format(path, interp, standardize=not is_mask,
                            is_mask=is_mask, target_size=tiling_size,
                            imagenet_norm=imagenet_norm)
    if not is_mask and p.c['preprocess'].get('collapse', False):
        arr = p.collapse_channels(arr)

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
        eps = 1e-6
        intersection = (pred_map * mask_stitched).sum()
        union = pred_map.sum() + mask_stitched.sum() - intersection
        iou_val = ((intersection + eps) / (union + eps)).item()
        acc_val = (pred_map == mask_stitched).float().mean().item()

        axes[3].imshow(mask_stitched.numpy(), cmap='gray', vmin=0, vmax=1)
        axes[3].set_title(f'GT  |  IoU={iou_val:.3f}  Acc={acc_val:.3f}')

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
    parser.add_argument('-c', '--config', default='config.json',
                        help='config JSON with preprocess section (default: config.json)')
    parser.add_argument('--mask',         default=None,  help='ground-truth mask TIFF (single-image mode)')
    parser.add_argument('--output',       default=None,  help='save figure here instead of displaying')
    parser.add_argument('--threshold',    default=0.5, type=float,
                        help='sigmoid threshold for binary prediction (default 0.5)')
    main(parser.parse_args())
