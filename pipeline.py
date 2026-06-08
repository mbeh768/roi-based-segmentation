"""
End-to-end segmentation + distance-band analysis for a single .czi slide.

Everything is driven by a pipeline.json config file (no CLI flags). Preprocess/tiling
params come from the config embedded in the checkpoint (so they always match the model;
pipeline.json "config" is an optional override). Given a model checkpoint and one .czi
this script:

  1. Prompts the user to draw an ROI, then runs the segmentation (tiles forwarded at the
     model's trained tiling_size).
  2. Crops the zero-padding the preprocessing introduced from the predicted mask.
  3. Upsamples the cropped mask to the slide pixel grid (aspect ratio preserved); the
     working resolution is chosen so the output's longest side ~= target_longest_side.
  4. Saves an overlay of the mask on the image at alpha 0.5  (mask_overlay.png).
  5. Reads the slide's physical pixel size (microns/pixel) from the CZI scaling metadata
     and shades the foreground (band 0) plus sequential outward distance bands whose
     thicknesses are given by the `bands` array, colored with a cycling palette
     (band_overlay.png).

Usage:
    python pipeline.py                # reads ./pipeline.json
    python pipeline.py my_run.json    # explicit config path

See pipeline.json for the available keys.
"""
import argparse
import colorsys
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch

import model.model as module_arch  # noqa: F401  (kept for parity / future use)
from inference import load_model, predict_prob_map
from preprocess import Preprocessor
from annotate_globals import (BrushROIDrawer, _apply_gamma, _draw_banner,
                              _fill_holes)


# ── config ────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "target_longest_side": 8000,  # zoom is auto-picked so the output's longest side ~= this (memory-safe on huge slides)
    "threshold": 0.5,
    "scene": 0,
    "microns_per_pixel": None,  # override the slide's NATIVE um/px (None -> read metadata)
    "keep_connected": 0,        # keep only the N largest foreground blobs (0/null -> keep all)
    "bands": [100],             # sequential band thicknesses (microns) outward from fg, e.g.
                                #   [100,100,100,100,100] -> 5 bands of 100um;  [10,20,30] -> bands of 10,20,30um
    "palette_size": 8,
    "outdir": None,             # None -> saved/pipeline/<stem>/
    "save_max": 8192,           # cap longest side of SAVED pngs
    "alpha": 0.5,
    "roi_canvas_size": 768,     # interactive ROI drawing window size (px)
    "roi_brush_radius": 8,      # initial ROI brush radius (canvas px)
}


def load_pipeline_config(path):
    with open(path) as f:
        pp = json.load(f)
    for required in ("model", "image"):
        if required not in pp:
            raise KeyError(f"pipeline config '{path}' is missing required key '{required}'")
    cfg = dict(DEFAULTS)
    cfg.update(pp)
    if not cfg.get("bands"):
        raise ValueError("pipeline config 'bands' must be a non-empty list of band widths (microns)")
    return cfg


# ── czi metadata ────────────────────────────────────────────────────────────────
def czi_native_dims_and_scale(path, scene_index=0):
    """Return (H, W, um_x, um_y) for a CZI: native pixel dims and microns/pixel.

    um_x/um_y come from Scaling > Items > Distance (stored in meters), or None if the
    metadata is missing/unparseable.
    """
    from pylibCZIrw import czi as pyczi

    with pyczi.open_czi(str(path)) as czidoc:
        bb = czidoc.total_bounding_box
        H = bb["Y"][1] - bb["Y"][0]
        W = bb["X"][1] - bb["X"][0]
        md = czidoc.raw_metadata

    um_x = um_y = None
    try:
        root = ET.fromstring(md)
        for dist in root.findall(".//Scaling/Items/Distance"):
            val = dist.findtext("Value")
            if val is None:
                continue
            um = float(val) * 1e6  # meters -> microns
            if dist.get("Id") == "X":
                um_x = um
            elif dist.get("Id") == "Y":
                um_y = um
    except Exception as e:  # pragma: no cover - metadata variability
        print(f"warning: failed to parse CZI scaling metadata ({type(e).__name__}: {e})")
    return H, W, um_x, um_y


def load_display_image(p, img_path, scene_index, zoom):
    """Load the CZI at `zoom` and return an (H, W, 3) float32 RGB image in [0, 1]."""
    chw = p._load_czi(img_path, scene_index=scene_index, zoom=zoom)   # CHW, original dtype
    hwc = np.transpose(chw, (1, 2, 0)).astype(np.float32)
    if chw.dtype == np.uint8:
        hwc /= 255.0
    elif chw.dtype == np.uint16:
        hwc /= 65535.0
    else:
        vmin, vmax = float(hwc.min()), float(hwc.max())
        if vmax > vmin:
            hwc = (hwc - vmin) / (vmax - vmin)
    hwc = np.clip(hwc, 0.0, 1.0)

    C = hwc.shape[2]
    if C == 1:
        hwc = np.repeat(hwc, 3, axis=2)
    elif C > 3:
        hwc = hwc[:, :, :3]
    elif C == 2:
        hwc = np.concatenate([hwc, np.zeros_like(hwc[:, :, :1])], axis=2)
    return hwc


# ── interactive ROI ─────────────────────────────────────────────────────────────
def roi_canvas_bgr(p, img_path, scene, native_H, native_W, canvas_size):
    """Build a square BGR canvas that matches the model's padded-square global view.

    Replicates load_and_format's geometry (content scaled so longest side fits, placed
    top-left, the rest zero-padded) so an ROI drawn here aligns with the global image
    the model actually sees. Uses a cheap pyramid level for speed.
    """
    z = min(1.0, 2000.0 / max(native_H, native_W))
    rgb = load_display_image(p, img_path, scene, z)               # (h, w, 3) float [0,1]
    sc = canvas_size / max(native_H, native_W)
    cw, ch = max(1, int(native_W * sc)), max(1, int(native_H * sc))
    content = cv2.resize(rgb, (cw, ch), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((canvas_size, canvas_size, 3), np.float32)
    canvas[:ch, :cw] = content
    return cv2.cvtColor((np.clip(canvas, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def prompt_for_roi(canvas_bgr, target_size, brush_radius=8):
    """Open an interactive window to draw an ROI; return a (1, target, target) tensor.

    Returns None if the user cancels (q / ESC). Reuses annotate_globals' BrushROIDrawer.
    Keys: L-drag paint, Shift+L-drag erase, -/= brush, f flood-fill (y/n), c clear,
          [ / ] gamma, \\ reset gamma, s / Enter accept, q / ESC cancel.
    """
    cs = canvas_bgr.shape[0]
    drawer = BrushROIDrawer(radius=brush_radius)
    drawer.reset(cs, cs)
    win = "pipeline: draw ROI  (s/Enter=accept  c=clear  f=fill  -/= brush  q=cancel)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, cs, cs)
    cv2.setMouseCallback(win, lambda e, x, y, f, _u: drawer.on_mouse(e, x, y, f))

    gamma = 1.0
    pending_fill = None
    result = None
    try:
        while True:
            gc = _apply_gamma(canvas_bgr, gamma)
            if pending_fill is not None:
                saved = drawer.mask
                drawer.mask = pending_fill
                view = drawer.render_overlay(gc)
                drawer.mask = saved
                _draw_banner(view, ["FILL PREVIEW", "y = accept   n / ESC = reject"])
            else:
                view = drawer.render_overlay(gc)
                _draw_banner(view, [
                    "Draw ROI for the model",
                    "s/Enter accept  c clear  f fill  -/= brush  q cancel",
                    f"brush={drawer.radius}px  gamma={gamma:.2f}",
                ])
            cv2.imshow(win, view)
            key = cv2.waitKey(15) & 0xFF

            if pending_fill is not None:
                if key == ord('y'):
                    drawer.mask = pending_fill
                    pending_fill = None
                elif key in (ord('n'), 27):
                    pending_fill = None
                continue

            if key in (ord('q'), 27):
                result = None
                break
            elif key in (ord('s'), 13, 10):
                result = drawer.get_mask(target_size, target_size)
                break
            elif key == ord('c'):
                drawer.clear()
            elif key == ord('f'):
                pending_fill = _fill_holes(drawer.mask)
            elif key in (ord('-'), ord('_')):
                drawer.set_radius(drawer.radius - max(1, drawer.radius // 8))
            elif key in (ord('='), ord('+')):
                drawer.set_radius(drawer.radius + max(1, drawer.radius // 8))
            elif key in (ord('['), ord('{')):
                gamma = max(0.1, round(gamma - 0.25, 2))
            elif key in (ord(']'), ord('}')):
                gamma = min(8.0, round(gamma + 0.25, 2))
            elif key == ord('\\'):
                gamma = 1.0
    finally:
        cv2.destroyWindow(win)
    return result


def keep_largest_blobs(fg, n):
    """Keep only the `n` largest connected components of a boolean mask, dropping the
    rest. n in (None, 0, negative) -> keep everything. Returns (mask, n_removed)."""
    if not n or n <= 0:
        return fg, 0
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg.astype(np.uint8), connectivity=8)
    n_blobs = num - 1                                   # label 0 is background
    if n_blobs <= n:
        return fg, 0
    areas = stats[1:, cv2.CC_STAT_AREA]                 # area per blob (labels 1..num-1)
    keep = np.argsort(areas)[::-1][:n] + 1             # label ids of the n biggest
    out = np.isin(labels, keep)
    return out, n_blobs - n


# ── overlays ────────────────────────────────────────────────────────────────────
def cycling_palette(k):
    """k distinct RGB colors (float [0,1]) spread over the hue circle."""
    return np.array([colorsys.hsv_to_rgb(i / k, 0.9, 1.0) for i in range(k)],
                    dtype=np.float32)


def blend(base, color_rgb, mask, alpha):
    """Alpha-blend a flat color onto `base` (HWC float) wherever `mask` is True."""
    out = base.copy()
    out[mask] = (1.0 - alpha) * base[mask] + alpha * np.asarray(color_rgb, np.float32)
    return out


def save_overlay(arr01, path, save_max, text_lines=None):
    """Resize (if over save_max longest side), draw optional text, write as PNG."""
    h, w = arr01.shape[:2]
    longest = max(h, w)
    out = arr01
    if longest > save_max:
        s = save_max / longest
        out = cv2.resize(arr01, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor((np.clip(out, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if text_lines:
        y = 30
        for line in text_lines:
            cv2.putText(bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y += 28
    cv2.imwrite(str(path), bgr)
    print(f"saved -> {path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main(pp):
    img_path = Path(pp["image"])
    scene = int(pp["scene"])

    model, device, ckpt_config = load_model(pp["model"])

    # Preprocess/tiling params MUST match what the model was trained with, so default
    # to the config embedded in the checkpoint. pipeline.json "config" is an optional
    # override only (advanced use); mismatched tiling collapses the segmentation.
    if pp.get("config"):
        with open(pp["config"]) as f:
            cfg = json.load(f)
        print(f"using preprocess config from {pp['config']} (override)")
    else:
        cfg = ckpt_config.config
        print(f"using preprocess config embedded in checkpoint "
              f"(tiling_size={cfg['preprocess']['tiling_size']}, "
              f"tile_size={cfg['preprocess']['tile_size']})")
    p = Preprocessor(cfg)

    # native dims + physical scale
    H, W, um_x, um_y = czi_native_dims_and_scale(img_path, scene_index=scene)
    print(f"\nnative CZI dims: {H} x {W}  ({H * W / 1e6:.0f} MP)")

    if pp["microns_per_pixel"] is not None:
        native_um = float(pp["microns_per_pixel"])
        print(f"native microns/pixel: {native_um:.4f}  (override)")
    else:
        if um_x is None or um_y is None:
            raise ValueError(
                "could not read microns/pixel from CZI metadata; "
                "set 'microns_per_pixel' in the pipeline config."
            )
        if abs(um_x - um_y) / um_x > 0.01:
            print(f"WARNING: anisotropic scaling (X={um_x:.4f}, Y={um_y:.4f} um/px); "
                  f"using X. Set 'microns_per_pixel' to override.")
        native_um = um_x
        print(f"native microns/pixel: {native_um:.4f}  (from metadata)")

    # pick the pyramid zoom so the output's longest side ~= target_longest_side
    # (never upscale past native: zoom capped at 1.0)
    target = int(pp["target_longest_side"])
    z = min(1.0, target / max(H, W))
    print(f"target longest side: {target}px -> pyramid zoom: {z:.5f}")

    dt_gb = (H * z) * (W * z) * 4 / 1e9
    if dt_gb > 12:
        print(f"WARNING: distance transform at this size needs ~{dt_gb:.1f} GB RAM; "
              f"consider a smaller 'target_longest_side'.")

    work_um = native_um / z

    # always prompt for an ROI (drawn on the model's padded-square global view)
    g = int(cfg["preprocess"]["global_downsample"])
    print("\nopening interactive ROI window — draw the ROI, then press s/Enter to accept...")
    canvas = roi_canvas_bgr(p, img_path, scene, H, W, int(pp["roi_canvas_size"]))
    roi = prompt_for_roi(canvas, g, brush_radius=int(pp["roi_brush_radius"]))
    if roi is None:
        sys.exit("ROI drawing cancelled — aborting.")
    if not getattr(model, "use_roi", False):
        print("note: this model was not trained with use_roi; the drawn ROI will be ignored.")

    # 1-2. segment + crop padding
    prob_map, info = predict_prob_map(img_path, model, device, p, cfg,
                                      scene_index=scene, roi=roi)
    tsz = info["H"]                                  # tiling_size (padded square)
    scale = tsz / max(H, W)                          # matches load_and_format
    new_H, new_W = int(H * scale), int(W * scale)    # content region in the square
    prob_crop = prob_map[:new_H, :new_W].cpu().numpy().astype(np.float32)

    # 3. load display image at zoom, then upsample mask to its exact pixel grid
    disp = load_display_image(p, img_path, scene, z)   # (H_w, W_w, 3)
    H_w, W_w = disp.shape[:2]
    prob_up = cv2.resize(prob_crop, (W_w, H_w), interpolation=cv2.INTER_LINEAR)
    fg = prob_up > float(pp["threshold"])              # boolean foreground

    fg, n_removed = keep_largest_blobs(fg, pp.get("keep_connected"))
    if n_removed:
        print(f"keep_connected={pp['keep_connected']}: removed {n_removed} smaller blob(s)")

    print(f"work dims: {H_w} x {W_w}   working microns/pixel: {work_um:.4f}")
    if not fg.any():
        print("WARNING: no foreground predicted above threshold — overlays will show no mask.")

    outdir = Path(pp["outdir"]) if pp["outdir"] else Path("saved/pipeline") / img_path.stem
    outdir.mkdir(parents=True, exist_ok=True)
    alpha = float(pp["alpha"])
    save_max = int(pp["save_max"])

    # 4. mask overlay (alpha 0.5)
    mask_overlay = blend(disp, (1.0, 1.0, 1.0), fg, alpha)
    save_overlay(mask_overlay, outdir / "mask_overlay.png", save_max)

    # 5. sequential distance bands outward from the foreground. Each entry in `bands`
    #    is a band thickness in microns; cumulative boundaries define where each band
    #    starts/ends. Banding stops at the sum of the thicknesses (tissue beyond is
    #    left uncolored).
    bands_um = [float(b) for b in pp["bands"]]
    boundaries_um = np.cumsum([0.0] + bands_um)          # len(bands)+1 edges, microns
    dist_px = cv2.distanceTransform((~fg).astype(np.uint8), cv2.DIST_L2, 5)
    dist_um = dist_px * work_um
    # color index per pixel: 0 = foreground (the 0um band), 1..len = the outward distance
    # bands defined by `bands`, -1 = tissue beyond the last boundary (uncolored).
    ci = np.digitize(dist_um, boundaries_um).astype(np.int32)   # 1..len(bands)+1 for non-fg
    ci[ci > len(bands_um)] = -1                                 # beyond last boundary
    ci[fg] = 0                                                  # foreground is band 0
    colored = ci >= 0
    counts = [int((ci == i).sum()) for i in range(len(bands_um) + 1)]
    print(f"bands: foreground=band0, then widths(um)={bands_um}  "
          f"boundaries(um)={[round(float(b),1) for b in boundaries_um]}  px-counts(band0..N)={counts}")

    k = max(1, int(pp["palette_size"]))
    palette = cycling_palette(k)
    band_rgb = np.zeros_like(disp)
    band_rgb[colored] = palette[ci[colored] % k]
    band_rgb[ci == 0] = (1.0, 1.0, 1.0)            # foreground (band 0) shaded white
    band_overlay = disp.copy()
    band_overlay[colored] = (1.0 - alpha) * disp[colored] + alpha * band_rgb[colored]

    text = [
        f"{work_um:.3f} um/px   band0=fg  bands(um)={[int(b) for b in bands_um]}   total {boundaries_um[-1]:.0f}um",
        f"native {H}x{W}  zoom {z:.4f}",
    ]
    save_overlay(band_overlay, outdir / "band_overlay.png", save_max, text_lines=text)

    print(f"\ndone -> {outdir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", default="pipeline.json",
                        help="path to pipeline config json (default: pipeline.json)")
    args = parser.parse_args()
    if not Path(args.config).exists():
        sys.exit(f"pipeline config not found: {args.config}")
    main(load_pipeline_config(args.config))
