"""
Interactive one-shot demo: pick a TIFF, draw an ROI if the model needs one,
run inference, see the result.

Prompts for:
  - path to a TIFF image (defaults to a held-out AJKY121_8 pair)
  - model checkpoint (defaults to the current best experiment)
  - config file (defaults to config.json)

If the checkpoint was trained with use_roi=True, opens an interactive brush
window (same drawing UI as annotate_globals.py) over the downsampled global
view of the image so you can paint the ROI before inference runs. The drawn
ROI is also saved to demo_rois/<stem>_roi.pt so re-running on the same image
doesn't require redrawing (answer 'n' when asked to reuse it if you want a
fresh draw).

Usage:
    python demo.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from annotate_globals import BrushROIDrawer, _apply_gamma, _draw_banner, _fill_holes, _global_to_bgr
from inference import load_model, run_one, tif_to_tensor
from preprocess import Preprocessor

# _6 is the specific fold that verifiably excludes AJKY121-8-1/2 from both its
# train and val sets (see verification/gather_evidence.py + evidence/AJKY121-8.json)
# -- the plain (unsuffixed) run trains on ~70% of this image's own tiles, so it
# would not be a genuine held-out demo for this case.
DEFAULT_MODEL = 'saved/models/EXPERIMENT 10A_old [wholistic performance evaluation]_6/model_best.pth'
DEFAULT_CONFIG = 'config.json'
# One of the held-out pairs used to generate the quant/learned predictions.
DEFAULT_IMAGE = ("/mnt/2TBhdd/mayo_backups/Max B_files_from Behfar_Shared folder(formatted)/"
                  "Segmentation_Project/RGB FULL - dataset/AJKY121_8/AJKY121-8-1_image.tif")
ROI_DIR = Path('demo_rois')


def prompt(msg, default=None):
    suffix = f" [{default}]" if default else ""
    val = input(f"{msg}{suffix}: ").strip()
    # Strip a single matching pair of quotes -- terminals that quote paths on
    # drag-and-drop (or a copy-pasted `'...'`/`"..."` argument) leave the quote
    # characters in the string since input() isn't a shell and won't do it.
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
        val = val[1:-1]
    return val or default


def prompt_yes_no(msg, default=True):
    suffix = " [Y/n]" if default else " [y/N]"
    val = input(f"{msg}{suffix}: ").strip().lower()
    if not val:
        return default
    return val.startswith('y')


def draw_roi(image_path, config, canvas_size=768, brush_radius=8):
    """Open an interactive brush window over the model's own global (downsampled)
    view of the image and return a (1, H, W) float32 ROI tensor at that same
    resolution -- the resolution the model's ROI input branch expects (it's fed
    alongside the global tensor itself, see predict_prob_map in inference.py).
    Returns None if the user cancels."""
    glob = tif_to_tensor(image_path, config, is_mask=False)   # (C, H, W)
    _, H, W = glob.shape
    bgr = _global_to_bgr(glob)
    canvas = cv2.resize(bgr, (canvas_size, canvas_size), interpolation=cv2.INTER_LINEAR)

    drawer = BrushROIDrawer(radius=brush_radius)
    drawer.reset(canvas_size, canvas_size)

    win = 'draw ROI'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, canvas_size, canvas_size)
    cv2.setMouseCallback(win, lambda e, x, y, f, _u: drawer.on_mouse(e, x, y, f))
    cv2.createTrackbar('brush', win, brush_radius, 50, lambda v: drawer.set_radius(max(1, v)))

    print("Draw the ROI: left-drag to paint, shift+left-drag to erase, "
          "'-'/'=' to resize brush, 'f' to flood-fill enclosed regions, "
          "'['/']' to darken/brighten (\\\\ resets), 's' to accept, 'c' to clear, 'q'/ESC to cancel.")
    accepted = False
    gamma = 1.0
    pending_fill = None  # uint8 mask awaiting y/n confirmation
    while True:
        gamma_canvas = _apply_gamma(canvas, gamma)

        if pending_fill is not None:
            left = gamma_canvas.copy()
            saved_mask = drawer.mask
            drawer.mask = pending_fill
            right = drawer.render_overlay(gamma_canvas)
            drawer.mask = saved_mask
            view = np.hstack([left, right])
            _draw_banner(view, ['FILL PREVIEW', 'y = accept   n / ESC = reject'])
            cv2.imshow(win, view)
            key = cv2.waitKey(15) & 0xFF
            if key == ord('y'):
                drawer.mask = pending_fill
                pending_fill = None
            elif key in (ord('n'), 27):
                pending_fill = None
            continue

        view = drawer.render_overlay(gamma_canvas)
        cv2.putText(view, f"s=accept  c=clear  f=fill  [/]=gamma({gamma:.2f})  q=cancel", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(win, view)
        key = cv2.waitKey(15) & 0xFF
        if key == ord('s'):
            accepted = True
            break
        if key == ord('c'):
            drawer.clear()
        elif key == ord('f'):
            pending_fill = _fill_holes(drawer.mask)
        elif key in (ord('['), ord('{')):
            gamma = max(0.1, round(gamma - 0.25, 2))
        elif key in (ord(']'), ord('}')):
            gamma = min(8.0, round(gamma + 0.25, 2))
        elif key == ord('\\'):
            gamma = 1.0
        elif key in (ord('-'), ord('_')):
            drawer.set_radius(drawer.radius - max(1, drawer.radius // 8))
            cv2.setTrackbarPos('brush', win, drawer.radius)
        elif key in (ord('='), ord('+')):
            drawer.set_radius(drawer.radius + max(1, drawer.radius // 8))
            cv2.setTrackbarPos('brush', win, drawer.radius)
        elif key in (ord('q'), 27):
            break
    cv2.destroyWindow(win)
    if not accepted:
        return None
    return drawer.get_mask(H, W)


def main():
    print("=== att_vent_seg demo ===\n")

    image_path = prompt("Path to TIFF image", DEFAULT_IMAGE)
    while not image_path or not Path(image_path).exists():
        print(f"  file not found: {image_path!r}")
        image_path = prompt("Path to TIFF image", DEFAULT_IMAGE)
    image_path = Path(image_path)

    model_path = prompt("Model checkpoint", DEFAULT_MODEL)
    while not Path(model_path).exists():
        print(f"  checkpoint not found: {model_path!r}")
        model_path = prompt("Model checkpoint", DEFAULT_MODEL)

    config_path = prompt("Config file", DEFAULT_CONFIG)
    while not Path(config_path).exists():
        print(f"  config not found: {config_path!r}")
        config_path = prompt("Config file", DEFAULT_CONFIG)

    print(f"\nLoading model from {model_path} ...")
    model, device, _ = load_model(model_path)

    with open(config_path) as f:
        config = json.load(f)
    p = Preprocessor(config)

    roi_tensor = None
    if getattr(model, 'use_roi', False):
        stem = image_path.stem.replace('_image', '').replace('_mask', '')
        roi_cache = ROI_DIR / f'{stem}_roi.pt'
        if roi_cache.exists() and prompt_yes_no(f"\nFound a previously-drawn ROI at {roi_cache} -- reuse it?"):
            roi_tensor = torch.load(roi_cache, weights_only=True)
        else:
            print("\nThis model requires an ROI. Opening the ROI drawing window...")
            roi_tensor = draw_roi(image_path, config)
            if roi_tensor is None:
                print("ROI drawing cancelled -- aborting.")
                return
            ROI_DIR.mkdir(exist_ok=True)
            torch.save(roi_tensor, roi_cache)
            print(f"  saved ROI -> {roi_cache}")
    else:
        print("\nThis model does not require an ROI; skipping.")

    print("\nRunning inference...")
    run_one(image_path, None, model, device, p, threshold=0.5, title=image_path.name,
            roi=roi_tensor, config=config)


if __name__ == '__main__':
    main()
