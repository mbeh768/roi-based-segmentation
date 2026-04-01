"""
Run global inference on a raw TIFF image using a trained SimpleUNet.
Preprocesses the TIFF on the fly (no files saved) using the same pipeline as preprocess.py.

Usage:
    # single image
    python inference_global.py --image slide.tif -m saved/models/.../model_best.pth
    python inference_global.py --image slide.tif --mask slide_mask.tif -m <ckpt>
    python inference_global.py --image slide.tif -m <ckpt> --output pred.png

    # whole dataset from config's raw_image_directory (close window to advance)
    python inference_global.py --dataset-pairs -m <ckpt>
"""
import argparse
import random

import cv2
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
    arr = p.load_and_format(path, interp, standardize=not is_mask, is_mask=is_mask, imagenet_norm=imagenet_norm)
    return torch.from_numpy(arr)   # (C, H, W) float32


def to_display(tensor):
    """(C, H, W) float32 → (H, W, C) uint8. Matches Preprocessor._make_vis."""
    return Preprocessor(None)._make_vis(tensor.cpu().float().numpy())


def run_one(img_path, mask_path, model, device, config, threshold, title, output=None):
    print(f"\nPreprocessing {img_path.name}...")
    image = tif_to_tensor(img_path, config, is_mask=False)

    mask = None
    if mask_path is not None:
        print(f"Preprocessing mask {mask_path.name}...")
        mask = tif_to_tensor(mask_path, config, is_mask=True)

    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
    probs = torch.sigmoid(logits).squeeze().cpu()
    pred  = (probs > threshold).float()

    n_cols = 4 if mask is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    fig.suptitle(title)

    axes[0].imshow(to_display(image))
    axes[0].set_title('Input')

    axes[1].imshow(probs.numpy(), cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Probability')

    axes[2].imshow(pred.numpy(), cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'Prediction (t={threshold})')

    if mask is not None:
        axes[3].imshow(mask.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[3].set_title('Ground truth')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved → {output}")
        plt.close(fig)
    else:
        plt.show()


def main(args):
    model, device, config = load_model(args.model)

    if args.dataset_pairs:
        p = Preprocessor(config)
        image_paths, mask_paths = p.find_source_filepaths()
        pairs = list(zip(image_paths, mask_paths))
        random.shuffle(pairs)
        print(f"Found {len(pairs)} image/mask pairs")
        for i, (img_path, mask_path) in enumerate(pairs):
            title = f"{img_path.name}  [{i+1}/{len(pairs)}]"
            run_one(img_path, mask_path, model, device, config, args.threshold, title, args.output)
    else:
        from pathlib import Path
        img_path  = Path(args.image)
        mask_path = Path(args.mask) if args.mask else None
        run_one(img_path, mask_path, model, device, config, args.threshold, img_path.name, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--image',         help='path to a single input TIFF')
    src.add_argument('--dataset-pairs', action='store_true',
                     help='loop through all image/mask pairs from config raw_image_directory')

    parser.add_argument('-m', '--model',  required=True, help='path to checkpoint (.pth)')
    parser.add_argument('--mask',         default=None,  help='ground-truth mask TIFF (single-image mode)')
    parser.add_argument('--output',       default=None,  help='save figure here instead of displaying')
    parser.add_argument('--threshold',    default=0.5, type=float,
                        help='sigmoid threshold for binary prediction (default 0.5)')
    main(parser.parse_args())
