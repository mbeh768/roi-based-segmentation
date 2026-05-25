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


def _resolve_roi(roi_arg, img_path, config):
    """Return a (1, H, W) float32 ROI tensor, or None if not required."""
    if roi_arg is not None:
        return torch.load(Path(roi_arg), weights_only=True)
    # Auto-lookup: data_dir/images/<stem>/roi_<stem>.pt, matching preprocess layout.
    try:
        data_dir = Path(config['data_loader']['args']['data_dir'])
    except (KeyError, TypeError):
        return None
    stem = img_path.stem.replace('_image', '').replace('_mask', '')
    candidate = data_dir / 'images' / stem / f'roi_{stem}.pt'
    if candidate.exists():
        return torch.load(candidate, weights_only=True)
    return None


def run_one(img_path, mask_path, model, device, p, threshold, title, output=None,
            roi_path=None, config=None):
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

    model_args = [local_batch, glob_batch, pos_batch]
    roi = None
    if getattr(model, 'use_roi', False):
        roi = _resolve_roi(roi_path, img_path, config)
        if roi is None:
            raise FileNotFoundError(
                f"model has use_roi=True but no ROI found for {img_path.name}. "
                f"Pass --roi <path/to/roi_<stem>.pt> or run annotate_globals.py."
            )
        roi_batch = roi.unsqueeze(0).expand(N, -1, -1, -1).to(device)  # (N, roi_ch, gH, gW)
        model_args.append(roi_batch)

    with torch.no_grad():
        logits = model(*model_args)                               # (N, 1, tile_H, tile_W)
    probs_tiles = torch.sigmoid(logits).squeeze(1).cpu()           # (N, tile_H, tile_W)

    dl_args = (config or {}).get('data_loader', {}).get('args', {})
    if roi is not None and dl_args.get('roi_exclusion', False):
        radius = dl_args.get('roi_exclusion_radius', 0.05)
        roi_bin = (roi.squeeze(0).numpy() > 0.5).astype(np.uint8)
        H_roi, W_roi = roi_bin.shape
        dist = cv2.distanceTransform(1 - roi_bin, cv2.DIST_L2, 5)
        threshold_roi_px = radius * min(H, W) * (H_roi / H)
        for i, (ty, tx) in enumerate([(y, x) for y in ys for x in xs]):
            cy_roi = int(np.clip((ty + tile_size / 2) / H * H_roi, 0, H_roi - 1))
            cx_roi = int(np.clip((tx + tile_size / 2) / W * W_roi, 0, W_roi - 1))
            if dist[cy_roi, cx_roi] > threshold_roi_px:
                probs_tiles[i] = 0.0

    prob_map = stitch(list(probs_tiles), ys, xs, H, W, tile_size, stride)
    pred_map = (prob_map > threshold).float()

    # ── figure ───────────────────────────────────────────────────────────────
    panels = [('input', None), ('prob', None), ('pred', None)]
    if roi is not None:
        panels.insert(1, ('roi', None))
    if mask_stitched is not None:
        panels.append(('gt', None))
    n_cols = len(panels)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    fig.suptitle(title)
    ax_by = dict(zip([k for k, _ in panels], axes))

    ax_by['input'].imshow(to_display(torch.from_numpy(full_arr)))
    ax_by['input'].set_title('Input')

    if 'roi' in ax_by:
        roi_np = roi.squeeze(0).cpu().numpy()
        roi_up = cv2.resize(roi_np, (W, H), interpolation=cv2.INTER_NEAREST)
        input_rgb = to_display(torch.from_numpy(full_arr))
        ax_by['roi'].imshow(input_rgb)
        ax_by['roi'].imshow(roi_up, cmap='Oranges', alpha=0.4, vmin=0, vmax=1)
        ax_by['roi'].set_title(f'ROI overlay  ({roi_np.shape[0]}x{roi_np.shape[1]} → {H}x{W})')

    ax_by['prob'].imshow(prob_map.numpy(), cmap='hot', vmin=0, vmax=1)
    ax_by['prob'].set_title('Probability')

    ax_by['pred'].imshow(pred_map.numpy(), cmap='gray', vmin=0, vmax=1)
    ax_by['pred'].set_title(f'Prediction (t={threshold})')

    if mask_stitched is not None:
        eps = 1e-6
        intersection = (pred_map * mask_stitched).sum()
        union = pred_map.sum() + mask_stitched.sum() - intersection
        iou_val = ((intersection + eps) / (union + eps)).item()
        acc_val = (pred_map == mask_stitched).float().mean().item()

        ax_by['gt'].imshow(mask_stitched.numpy(), cmap='gray', vmin=0, vmax=1)
        ax_by['gt'].set_title(f'GT  |  IoU={iou_val:.3f}  Acc={acc_val:.3f}')

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
        if args.roi is not None:
            print('WARNING: --roi is ignored in --dataset-pairs mode; '
                  'per-image ROIs are auto-resolved from data_dir/images/<stem>/roi_<stem>.pt')
        image_paths, mask_paths = p.find_source_filepaths()
        pairs = list(zip(image_paths, mask_paths))
        random.shuffle(pairs)
        print(f"Found {len(pairs)} image/mask pairs")
        uses_roi = getattr(model, 'use_roi', False)
        for i, (img_path, mask_path) in enumerate(pairs):
            if uses_roi and _resolve_roi(None, img_path, raw_config) is None:
                print(f"  [{i+1}/{len(pairs)}] {img_path.name}: no ROI found — skipping")
                continue
            title = f"{img_path.name}  [{i+1}/{len(pairs)}]"
            run_one(img_path, mask_path, model, device, p, args.threshold, title,
                    args.output, roi_path=None, config=raw_config)
    else:
        img_path  = Path(args.image)
        mask_path = Path(args.mask) if args.mask else None
        run_one(img_path, mask_path, model, device, p, args.threshold, img_path.name,
                args.output, roi_path=args.roi, config=raw_config)


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
    parser.add_argument('--roi',          default=None,
                        help='path to roi_<stem>.pt (required if model was trained with use_roi=true; '
                             'if omitted, inference will try data_dir/images/<stem>/roi_<stem>.pt)')
    parser.add_argument('--output',       default=None,  help='save figure here instead of displaying')
    parser.add_argument('--threshold',    default=0.5, type=float,
                        help='sigmoid threshold for binary prediction (default 0.5)')
    main(parser.parse_args())
