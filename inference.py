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
from matplotlib.patches import Rectangle

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


def tif_to_tensor(path, config, is_mask=False, scene_index=0):
    """Load a TIFF and return a (C, H, W) float32 tensor via the preprocess pipeline."""
    p = Preprocessor(config)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    imagenet_norm = config['preprocess'].get('imagenet_norm', False) if not is_mask else False
    arr = p.load_and_format(path, interp, standardize=not is_mask,
                            is_mask=is_mask, imagenet_norm=imagenet_norm,
                            scene_index=scene_index)
    if not is_mask and config['preprocess'].get('collapse', False):
        arr = p.collapse_channels(arr)
    return torch.from_numpy(arr)


def to_display(tensor):
    """(C, H, W) float32 → (H, W, C) uint8. Matches Preprocessor._make_vis."""
    return Preprocessor(None)._make_vis(tensor.cpu().float().numpy())


def tif_to_tiles(path, p, is_mask=False, scene_index=0):
    """Load TIFF at tiling_size resolution and cut into tiles.

    Image tiles are extracted at context_size (centered on the inner tile_size region);
    mask tiles at tile_size. Sampling that runs off the canvas is zero-padded.
    """
    tile_size    = p.c['preprocess']['tile_size']
    stride       = p.c['preprocess']['stride']
    tiling_size  = p.c['preprocess']['tiling_size']
    context_size = p.c['preprocess'].get('context_size', tile_size)

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    imagenet_norm = p.c['preprocess'].get('imagenet_norm', False)
    arr = p.load_and_format(path, interp, standardize=not is_mask,
                            is_mask=is_mask, target_size=tiling_size,
                            imagenet_norm=imagenet_norm, scene_index=scene_index)
    if not is_mask and p.c['preprocess'].get('collapse', False):
        arr = p.collapse_channels(arr)

    img_hwc = np.transpose(arr, (1, 2, 0))
    H, W = img_hwc.shape[:2]
    C = img_hwc.shape[2]

    save_size = tile_size if is_mask else context_size
    pad = (save_size - tile_size) // 2

    ys = list(range(0, H - tile_size + 1, stride))
    xs = list(range(0, W - tile_size + 1, stride))

    tiles = []
    for y in ys:
        for x in xs:
            cy, cx = y - pad, x - pad
            patch_hwc = np.zeros((save_size, save_size, C), dtype=np.float32)
            src_y0 = max(0, cy);  src_y1 = min(H, cy + save_size)
            src_x0 = max(0, cx);  src_x1 = min(W, cx + save_size)
            dst_y0 = src_y0 - cy; dst_y1 = src_y1 - cy
            dst_x0 = src_x0 - cx; dst_x1 = src_x1 - cx
            if src_y1 > src_y0 and src_x1 > src_x0:
                patch_hwc[dst_y0:dst_y1, dst_x0:dst_x1] = img_hwc[src_y0:src_y1, src_x0:src_x1]
            tiles.append(torch.from_numpy(np.transpose(patch_hwc, (2, 0, 1)).copy()))

    return tiles, ys, xs, H, W, tile_size, stride, arr


def stitch(tile_maps, ys, xs, H, W, tile_size, stride):
    """Average-stitch single-channel (H,W) tile predictions into a full canvas."""
    canvas = torch.zeros(H, W)
    count  = torch.zeros(H, W)
    for i, (y, x) in enumerate([(y, x) for y in ys for x in xs]):
        canvas[y:y+tile_size, x:x+tile_size] += tile_maps[i]
        count[y:y+tile_size,  x:x+tile_size] += 1
    return canvas / count.clamp(min=1)


def _resolve_roi(roi_arg, img_path, config, roi_idx=None):
    """Return a (1, H, W) float32 ROI tensor, or None if not required.

    roi_idx selects among multiple annotated ROIs for the same slide:
        0  -> roi_<stem>.pt  (primary)
        1  -> roi_<stem>_1.pt
        N  -> roi_<stem>_N.pt
    If roi_idx is None the primary is used (legacy behaviour).
    """
    if roi_arg is not None:
        return torch.load(Path(roi_arg), weights_only=True)
    # Auto-lookup under data_dir/images/<stem>/.
    try:
        data_dir = Path(config['data_loader']['args']['data_dir'])
    except (KeyError, TypeError):
        return None
    stem = img_path.stem.replace('_image', '').replace('_mask', '')
    suffix = '' if (roi_idx is None or roi_idx == 0) else f'_{roi_idx}'
    candidate = data_dir / 'images' / stem / f'roi_{stem}{suffix}.pt'
    if candidate.exists():
        return torch.load(candidate, weights_only=True)
    return None


def predict_prob_map(img_path, model, device, p, config, roi_path=None, roi_idx=None,
                     scene_index=0, roi=None):
    """Run the model over all tiles of an image and stitch the probabilities.

    This is the matplotlib-free core of ``run_one`` so it can be reused by other
    tools (e.g. pipeline.py).

    Args:
        roi (torch.Tensor|None): an explicit (1, H, W) float32 ROI to feed the model,
            bypassing the saved-file lookup. Only used when the model has use_roi=True.
            When None, the ROI is auto-resolved from roi_path / data_dir (legacy).

    Returns:
        prob_map (torch.Tensor): (H, W) sigmoid probabilities on the tiling-size
            square canvas (H == W == tiling_size, content in the top-left
            new_H x new_W region with the rest zero-padded).
        info (dict): plotting/analysis context with keys ``full_arr`` (CHW float32
            of the input rendered at tiling_size), ``tile_coords``, ``keep_idx``,
            ``tile_size``, ``H``, ``W`` and ``roi`` (or None).
    """
    print(f"\nPreprocessing {img_path.name}...")
    tiles, ys, xs, H, W, tile_size, stride, full_arr = tif_to_tiles(
        img_path, p, is_mask=False, scene_index=scene_index)
    glob = tif_to_tensor(img_path, p.c, is_mask=False,
                         scene_index=scene_index)         # (3, global_H, global_W)

    N = len(tiles)
    local_all = torch.stack(tiles)                                 # (N, 3, tile_H, tile_W) on CPU

    # pos spans the full context window (matches LocalGlobalDataset); reduces to
    # the inner tile when context_size == tile_size.
    context_size = p.c['preprocess'].get('context_size', tile_size)
    pad = (context_size - tile_size) // 2
    pos_all = torch.stack([
        torch.tensor([(y - pad) / H, (x - pad) / W,
                      (y - pad + context_size) / H, (x - pad + context_size) / W],
                     dtype=torch.float32)
        for y in ys for x in xs
    ])                                                              # (N, 4) on CPU

    if getattr(model, 'use_roi', False):
        if roi is None:
            roi = _resolve_roi(roi_path, img_path, config, roi_idx=roi_idx)
        if roi is None:
            raise FileNotFoundError(
                f"model has use_roi=True but no ROI found for {img_path.name}. "
                f"Pass --roi <path/to/roi_<stem>.pt> or run annotate_globals.py."
            )
    else:
        roi = None

    # Build a keep-mask over tiles from ROI exclusion (matches dataset filter).
    dl_args = (config or {}).get('data_loader', {}).get('args', {})
    tile_coords = [(y, x) for y in ys for x in xs]
    keep_idx = list(range(N))
    if roi is not None and dl_args.get('roi_exclusion', False):
        radius = dl_args.get('roi_exclusion_radius', 0.05)
        roi_bin = (roi.squeeze(0).numpy() > 0.5).astype(np.uint8)
        H_roi, W_roi = roi_bin.shape
        if roi_bin.max() > 0:
            dist = cv2.distanceTransform(1 - roi_bin, cv2.DIST_L2, 5)
            threshold_roi_px = radius * min(H, W) * (H_roi / H)
            keep_idx = []
            for i, (ty, tx) in enumerate(tile_coords):
                y0_roi = int(np.clip(ty / H * H_roi, 0, H_roi - 1))
                x0_roi = int(np.clip(tx / W * W_roi, 0, W_roi - 1))
                y1_roi = int(np.clip((ty + tile_size) / H * H_roi, 0, H_roi - 1)) + 1
                x1_roi = int(np.clip((tx + tile_size) / W * W_roi, 0, W_roi - 1)) + 1
                if dist[y0_roi:y1_roi, x0_roi:x1_roi].min() <= threshold_roi_px:
                    keep_idx.append(i)
            print(f"  ROI exclusion: forwarding {len(keep_idx)}/{N} tiles")

    # Chunk the forward to bound GPU memory. Pull batch size from the supplied
    # config's data_loader.args.batch_size; default to 8 if missing.
    bs = int(((config or {}).get('data_loader', {}).get('args', {}).get('batch_size', 8)))
    bs = max(1, bs)

    save_size = context_size
    probs_tiles = torch.zeros(N, save_size, save_size)               # excluded tiles stay 0

    glob_gpu = glob.to(device)
    roi_gpu = roi.to(device) if roi is not None else None
    with torch.no_grad():
        for k in range(0, len(keep_idx), bs):
            batch_idx = keep_idx[k:k+bs]
            n = len(batch_idx)
            local_batch = local_all[batch_idx].to(device, non_blocking=True)
            glob_batch  = glob_gpu.unsqueeze(0).expand(n, -1, -1, -1)
            pos_batch   = pos_all[batch_idx].to(device, non_blocking=True)
            model_args  = [local_batch, glob_batch, pos_batch]
            if roi_gpu is not None:
                model_args.append(roi_gpu.unsqueeze(0).expand(n, -1, -1, -1))
            logits = model(*model_args)                            # (n, 1, ctx_H, ctx_W)
            probs_tiles[batch_idx] = torch.sigmoid(logits).squeeze(1).cpu()

    # If the model saw a context patch larger than the stitch tile, crop predictions
    # back to the centered tile_size region before stitching.
    if probs_tiles.shape[-1] != tile_size:
        off = (probs_tiles.shape[-1] - tile_size) // 2
        probs_tiles = probs_tiles[..., off:off + tile_size, off:off + tile_size]

    # Stitch only kept tiles so excluded-tile zeros don't dilute predictions in
    # overlap regions near the ROI exclusion boundary.
    prob_canvas = torch.zeros(H, W)
    prob_count  = torch.zeros(H, W)
    for i in keep_idx:
        ty, tx = tile_coords[i]
        prob_canvas[ty:ty+tile_size, tx:tx+tile_size] += probs_tiles[i]
        prob_count[ty:ty+tile_size,  tx:tx+tile_size] += 1
    prob_map = prob_canvas / prob_count.clamp(min=1)

    info = {
        'full_arr': full_arr,
        'tile_coords': tile_coords,
        'keep_idx': keep_idx,
        'tile_size': tile_size,
        'H': H,
        'W': W,
        'roi': roi,
    }
    return prob_map, info


def run_one(img_path, mask_path, model, device, p, threshold, title, output=None,
            roi_path=None, config=None, roi_idx=None, roi=None):
    prob_map, info = predict_prob_map(img_path, model, device, p, config,
                                      roi_path=roi_path, roi_idx=roi_idx, roi=roi)
    full_arr    = info['full_arr']
    tile_coords = info['tile_coords']
    keep_idx    = info['keep_idx']
    tile_size   = info['tile_size']
    H, W        = info['H'], info['W']
    roi         = info['roi']

    mask_stitched = None
    if mask_path is not None:
        print(f"Preprocessing mask {mask_path.name}...")
        m_tiles, *m_rest = tif_to_tiles(mask_path, p, is_mask=True)
        m_ys, m_xs, m_H, m_W, m_ts, m_st = m_rest[:6]
        mask_stitched = stitch([t.squeeze() for t in m_tiles], m_ys, m_xs, m_H, m_W, m_ts, m_st)

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
        roi_masked = np.ma.masked_where(roi_up < 0.5, roi_up)
        ax_by['roi'].imshow(roi_masked, cmap='Greens', alpha=0.6, vmin=0, vmax=1)
        ax_by['roi'].set_title('ROI overlay')

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

    kept_set = set(keep_idx)
    for name, ax in ax_by.items():
        ax.axis('off')
        if name != 'prob':
            continue
        for i, (ty, tx) in enumerate(tile_coords):
            color = 'cyan' if i in kept_set else 'red'
            ax.add_patch(Rectangle((tx, ty), tile_size, tile_size,
                                   fill=False, edgecolor=color,
                                   linewidth=0.5, alpha=0.6))

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

    save_dir = None
    if args.savefigs:
        ckpt_path = Path(args.model)
        # saved/models/<name>/<run_id>/... -> saved/log/<name>/<run_id>/inference
        log_root = Path(*['saved/log'] + list(ckpt_path.parts[2:-1]))
        save_dir = log_root / 'inference'
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.switch_backend('Agg')
        print(f"Saving figures to {save_dir}/")

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
            if uses_roi and _resolve_roi(None, img_path, raw_config, roi_idx=args.roi_idx) is None:
                print(f"  [{i+1}/{len(pairs)}] {img_path.name}: no ROI found — skipping")
                continue
            title = f"{img_path.name}  [{i+1}/{len(pairs)}]"
            out = str(save_dir / f"{img_path.stem}.png") if save_dir else args.output
            run_one(img_path, mask_path, model, device, p, args.threshold, title,
                    out, roi_path=None, config=raw_config, roi_idx=args.roi_idx)
    else:
        img_path  = Path(args.image)
        mask_path = Path(args.mask) if args.mask else None
        out = str(save_dir / f"{img_path.stem}.png") if save_dir else args.output
        run_one(img_path, mask_path, model, device, p, args.threshold, img_path.name,
                out, roi_path=args.roi, config=raw_config, roi_idx=args.roi_idx)


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
    parser.add_argument('--roi-idx',      default=None, type=int,
                        help='select which annotated ROI to use: 0=roi_<stem>.pt, 1=roi_<stem>_1.pt, etc. '
                             '(default: 0 / primary)')
    parser.add_argument('--output',       default=None,  help='save figure here instead of displaying')
    parser.add_argument('--savefigs',     action='store_true',
                        help='save all figures to saved/log/<name>/<run_id>/inference/ instead of displaying')
    parser.add_argument('--threshold',    default=0.5, type=float,
                        help='sigmoid threshold for binary prediction (default 0.5)')
    main(parser.parse_args())
