"""
Ventricle segmentation on a full-size histology TIFF.

Point it at a slide, paint a region of interest over the ventricle wall, and it
writes a full-resolution binary mask TIFF matching the input's pixel dimensions.

    python segment.py                          # opens a file picker
    python segment.py --image slide.tif
    python segment.py --image slide.tif --out mask.tiff --threshold 0.5

Why the ROI step exists: the model was trained with `use_roi=True`, so every
tile it saw was paired with a hand-drawn mask marking roughly where the
ventricle wall is. It expects the same at inference time -- see roi_brush.py.
A drawn ROI is cached next to the output so re-running a slide can skip it.

Everything about how the image is tiled and normalised (tile size, stride,
downsample factors) is read from the config embedded in the checkpoint, so the
weights file is self-describing -- there is no config.json to keep in sync.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch

from inference import load_model, predict_prob_map, tif_to_tensor
from preprocess import Preprocessor
from roi_brush import draw_roi

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / 'model_best.pth'


def native_dims(path):
    """Read (height, width) of a TIFF without decoding the whole image.

    Gigapixel slides are far too large to load just to ask their size, so this
    reads the series metadata only. Axis order varies between exports (YXS,
    CYX, plain YX), so prefer tifffile's axes labels and fall back to assuming
    the two largest dimensions are the spatial ones -- true for any histology
    slide, where the channel axis is 1, 3 or 4.
    """
    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0]
        shape = tuple(int(s) for s in series.shape)
        axes = getattr(series, 'axes', '') or ''

    if 'Y' in axes and 'X' in axes:
        return shape[axes.index('Y')], shape[axes.index('X')]
    if len(shape) == 2:
        return shape
    two_largest = sorted(range(len(shape)), key=lambda i: shape[i])[-2:]
    y_idx, x_idx = sorted(two_largest)
    return shape[y_idx], shape[x_idx]


ISLAND_FRAC = 0.05  # drop foreground blobs <= this fraction of the largest blob's area


def remove_small_islands(mask, frac=ISLAND_FRAC):
    """Drop foreground components whose area is <= frac * the largest component.

    The model scores tiles independently, so it can leave a scatter of small
    disconnected specks away from the ventricle wall. Anything a twentieth the
    size of the main region is noise rather than anatomy. Comparably large
    regions are kept, so slides where the wall is genuinely split into two
    pieces survive intact.

    Matches the "remove islands" definition used by the overlay viewer
    (quant/make_overlay_viewer.py): 8-connectivity, threshold relative to the
    single largest component.

    Args:
        mask: uint8 array, 0 = background, non-zero = foreground.
    Returns:
        (cleaned uint8 mask of the same shape, number of components removed)
    """
    binary = (mask > 0).astype(np.uint8)
    # The label image is the same pixel count as the slide, so its dtype costs
    # real memory: 16-bit halves it versus the 32-bit default (about 1 GB saved
    # on a 500-megapixel slide, which matters on an 8 GB laptop). A slide with
    # more than 65535 separate blobs falls back to 32-bit automatically.
    try:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8, ltype=cv2.CV_16U)
    except cv2.error:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8, ltype=cv2.CV_32S)
    del binary
    if num_labels <= 2:                     # background plus at most one blob
        return mask, 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = areas.max()
    cutoff = frac * largest
    # Build a lookup table over label ids instead of testing each pixel, so this
    # stays a single pass over a 500-megapixel image.
    keep = np.zeros(num_labels, dtype=np.uint8)
    keep[1:] = (areas > cutoff).astype(np.uint8)
    removed = int((areas <= cutoff).sum())
    return (keep[labels] * 255).astype(np.uint8), removed


def ask_reuse_roi(roi_cache):
    """Ask whether to reuse a previously drawn ROI. True = reuse, False = redraw.

    Defaults to reusing on a bare Enter, and also when there is no console to
    read from (piped or scripted runs). Redrawing overwrites the saved file, so
    the safe option is the one that happens by accident.
    """
    try:
        when = datetime.fromtimestamp(
            roi_cache.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
    except OSError:
        when = 'unknown date'

    print(f"\nThis slide already has a region of interest:")
    print(f"  {roi_cache.name}   (saved {when})")
    print("     u = use it again")
    print("     d = draw a new one  (replaces the saved file)")
    while True:
        try:
            answer = input("  Use the existing region? [U/d]: ").strip().lower()
        except EOFError:
            print("\n  (no console to read from -- using the existing region)")
            return True
        if answer in ('', 'u', 'use', 'y', 'yes'):
            return True
        if answer in ('d', 'draw', 'n', 'no', 'r', 'redraw'):
            return False
        print("  Please type u (use it) or d (draw a new one).")


def gui_available():
    """True if we can safely open an OpenCV window.

    Checked up front because OpenCV's Qt backend calls abort() when it cannot
    reach a display -- the process dies with a core dump that no try/except can
    intercept, so probing after the fact is not an option. Windows and macOS
    always have a window server available to a logged-in user; on Linux the
    session variables are the reliable signal.
    """
    if sys.platform in ('win32', 'darwin'):
        return True
    import os
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


def pick_file_dialog():
    """Open a native file picker. Returns a Path, or None if cancelled."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title='Select a slide image',
        filetypes=[('TIFF images', '*.tif *.tiff *.TIF *.TIFF'), ('All files', '*.*')],
    )
    root.destroy()
    return Path(path) if path else None


def save_preview(preview_path, global_tensor, roi, mask, threshold):
    """Write a 3-panel PNG (input | ROI | prediction) for a quick sanity check.

    Reviewing a gigapixel mask TIFF is awkward, so this gives an at-a-glance
    view of whether the ROI landed where it should and the mask looks sane.
    """
    from roi_brush import global_to_bgr

    # The model's global view (and the ROI drawn on it) is a zero-padded SQUARE
    # with the slide in the top-left corner, while the mask is the slide's true
    # aspect ratio. Resizing the mask to the full square would stretch it across
    # the padding and misalign it with the tissue, so crop to the content region
    # first and scale everything to that.
    base_sq = global_to_bgr(global_tensor)
    sh, sw = base_sq.shape[:2]
    H_nat, W_nat = mask.shape[:2]
    scale = max(sh, sw) / max(H_nat, W_nat)
    ch = min(sh, max(1, int(round(H_nat * scale))))
    cw = min(sw, max(1, int(round(W_nat * scale))))

    base = base_sq[:ch, :cw]
    h, w = base.shape[:2]

    roi_panel = base.copy()
    roi_sq = cv2.resize(roi.squeeze(0).cpu().numpy().astype(np.float32), (sw, sh),
                        interpolation=cv2.INTER_NEAREST)
    roi_np = roi_sq[:ch, :cw]
    roi_panel[roi_np > 0.5] = (
        0.55 * roi_panel[roi_np > 0.5] + 0.45 * np.array([0, 180, 255])
    ).astype(np.uint8)

    # Downsample the mask that was actually written, so the preview cannot
    # disagree with the saved file (e.g. still showing removed islands).
    pred = cv2.resize((mask > 0).astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST)
    pred_panel = base.copy()
    pred_panel[pred > 0] = (
        0.45 * pred_panel[pred > 0] + 0.55 * np.array([0, 255, 0])
    ).astype(np.uint8)

    labels = ['input', 'ROI you drew', f'prediction (t={threshold})']
    panels = []
    for img, label in zip([base, roi_panel, pred_panel], labels):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 0), -1)
        cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 1, cv2.LINE_AA)
        panels.append(img)
    cv2.imwrite(str(preview_path), np.hstack(panels))


def main():
    ap = argparse.ArgumentParser(
        description='Segment the ventricle wall in a full-size histology TIFF.')
    ap.add_argument('--image', default=None,
                    help='input TIFF. If omitted, a file picker opens.')
    ap.add_argument('--out', default=None,
                    help='output mask TIFF (default: <input>_mask.tiff beside the input)')
    ap.add_argument('--model', default=str(DEFAULT_MODEL),
                    help='checkpoint .pth (default: model_best.pth beside this script)')
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='probability cutoff for foreground (default 0.5)')
    ap.add_argument('--roi', default=None,
                    help='reuse a saved ROI .pt instead of drawing one')
    ap.add_argument('--redraw', action='store_true',
                    help='ignore any cached ROI and draw a new one')
    ap.add_argument('--keep-islands', action='store_true',
                    help='keep small disconnected specks instead of removing them')
    ap.add_argument('--no-preview', action='store_true',
                    help='skip writing the 3-panel preview PNG')
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"ERROR: model checkpoint not found at {model_path}\n"
                 f"Place model_best.pth next to segment.py, or pass --model <path>.")

    if args.image:
        image_path = Path(args.image)
    else:
        print('Opening file picker...')
        image_path = pick_file_dialog()
        if image_path is None:
            sys.exit('No file selected.')
    if not image_path.exists():
        sys.exit(f"ERROR: image not found: {image_path}")

    out_path = Path(args.out) if args.out else \
        image_path.with_name(f'{image_path.stem}_mask.tiff')
    roi_cache = image_path.with_name(f'{image_path.stem}_roi.pt')

    print(f"\nLoading model: {model_path.name}")
    model, device, cfg = load_model(str(model_path))
    if device.type == 'cuda':
        print(f"  running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        # Say why, because "it's slow" is the most likely thing a collaborator
        # reports back, and the reason is almost always this line.
        why = ('no NVIDIA GPU detected' if not torch.cuda.is_available()
               else 'CPU build of PyTorch installed')
        print(f"  running on CPU ({why}) -- expect several minutes per slide")
        print(f"  to switch to GPU, run reconfigure.bat (Windows) "
              f"or ./reconfigure.sh")
    # predict_prob_map calls config.get(...), which ConfigParser does not
    # implement -- hand it the plain dict the ConfigParser wraps.
    raw_cfg = cfg.config if hasattr(cfg, 'config') else cfg
    p = Preprocessor(raw_cfg)

    H_native, W_native = native_dims(image_path)
    print(f"\nSlide: {image_path.name}")
    print(f"  native size: {W_native} x {H_native} px "
          f"({H_native * W_native / 1e6:.0f} MP)")

    roi = None
    if args.roi:
        roi = torch.load(Path(args.roi), weights_only=True)
        print(f"  ROI: loaded from {args.roi}")
    elif roi_cache.exists():
        if args.redraw:
            print(f"  ROI: --redraw given, ignoring saved {roi_cache.name}")
        elif ask_reuse_roi(roi_cache):
            roi = torch.load(roi_cache, weights_only=True)
            print(f"  ROI: using saved {roi_cache.name}")
        else:
            print(f"  ROI: drawing a new one; {roi_cache.name} will be replaced")

    # Reading a gigapixel TIFF takes a while and produces no output of its own,
    # so say what is happening first -- otherwise the tool looks frozen between
    # choosing a file and the drawing window appearing.
    print('\nReading the slide... (large images can take a few minutes)', flush=True)
    global_tensor = tif_to_tensor(image_path, raw_cfg, is_mask=False)
    print('  done reading.', flush=True)

    if roi is None and getattr(model, 'use_roi', False):
        if not gui_available():
            sys.exit(
                "ERROR: this slide has no region of interest yet, and there is no\n"
                "graphical display available to draw one on.\n\n"
                "Run this on a computer with a desktop, or draw the region there\n"
                "once and pass the saved file with --roi <slide>_roi.pt"
            )
        print("\nThis model needs a region of interest.")
        print("  A window will open: paint over the ventricle wall, then press 's'.")
        print("  Hover any row in the 'controls' window to see what it does.")
        try:
            roi = draw_roi(global_tensor, slide_name=image_path.name)
        except cv2.error as exc:
            # OpenCV raises a fairly opaque error when it has no display to draw
            # into (a machine with no desktop session, or a headless OpenCV
            # build). Translate it into something actionable.
            sys.exit(
                f"ERROR: could not open the drawing window.\n"
                f"  {exc}\n\n"
                f"This usually means the computer has no graphical desktop, or\n"
                f"OpenCV was installed without GUI support. Either run this on a\n"
                f"machine with a desktop, or draw the region elsewhere and pass\n"
                f"it with --roi <file>.pt"
            )
        if roi is None:
            sys.exit('ROI drawing cancelled -- nothing was written.')
        torch.save(roi, roi_cache)
        print(f"  saved ROI -> {roi_cache.name}")

    print('\nSegmenting (this can take a few minutes on CPU)...')
    prob_map, _info = predict_prob_map(
        image_path, model, device, p, raw_cfg, roi=roi)

    # The model works on a zero-padded square canvas at the trained tiling size.
    # Crop back to the aspect-correct content region, then resample the
    # probabilities (not a binarised mask) up to native resolution so the
    # threshold is applied once, on smooth values, giving clean edges.
    tsz = prob_map.shape[0]
    scale = tsz / max(H_native, W_native)
    new_h, new_w = int(H_native * scale), int(W_native * scale)
    prob_crop = prob_map[:new_h, :new_w].cpu().numpy().astype(np.float32)
    prob_up = cv2.resize(prob_crop, (W_native, H_native), interpolation=cv2.INTER_LINEAR)
    mask = (prob_up > args.threshold).astype(np.uint8) * 255
    del prob_up, prob_crop          # ~2 GB of float32; free before labelling

    if not args.keep_islands:
        before = int((mask > 0).sum())
        mask, removed = remove_small_islands(mask, ISLAND_FRAC)
        after = int((mask > 0).sum())
        if removed:
            print(f"  noise removal: dropped {removed} island(s) "
                  f"<= {int(ISLAND_FRAC * 100)}% of the largest region "
                  f"({before - after:,} px)")
        else:
            print("  noise removal: no islands to remove")

    tifffile.imwrite(str(out_path), mask, compression='zlib')
    pct = 100.0 * (mask > 0).mean()
    print(f"\nDone.")
    print(f"  padded {tsz}x{tsz} -> content {new_h}x{new_w} -> native {W_native}x{H_native}")
    print(f"  foreground: {int((mask > 0).sum()):,} px ({pct:.2f}% of slide)")
    print(f"  mask -> {out_path}")

    if not args.no_preview:
        preview_path = out_path.with_name(f'{out_path.stem}_preview.png')
        try:
            save_preview(preview_path, global_tensor, roi, mask, args.threshold)
            print(f"  preview -> {preview_path}")
        except Exception as exc:                    # preview is a nicety, not the deliverable
            print(f"  (preview skipped: {exc})")


if __name__ == '__main__':
    main()
