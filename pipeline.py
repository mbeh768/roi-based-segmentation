"""
End-to-end segmentation + distance-band analysis for a single .czi slide.

Everything is driven by a pipeline.json config file (no CLI flags). Given a model
checkpoint, the training config (for preprocess/tiling params), and one .czi image
this script:

  1. Runs the LocalGlobalUNet segmentation (tiles forwarded at tiling_size, as trained).
  2. Crops the zero-padding the preprocessing introduced from the predicted mask.
  3. Upsamples the cropped mask to the slide pixel grid (aspect ratio preserved) at a
     chosen pyramid zoom, so it aligns perfectly with the displayed image.
  4. Saves an overlay of the mask on the image at alpha 0.5  (mask_overlay.png).
  5. Reads the slide's physical pixel size (microns/pixel) from the CZI scaling metadata
     and grows concentric bands (default 100 um wide) outward from the ventricle
     foreground, out to the image edge, colored with a cycling palette
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
    "pyramid": None,            # zoom factor 0<z<=1; None -> czi_max_px halving loop
    "threshold": 0.5,
    "scene": 0,
    "microns_per_pixel": None,  # override the slide's NATIVE um/px (None -> read metadata)
    "band_microns": 100.0,
    "palette_size": 8,
    "outdir": None,             # None -> saved/pipeline/<stem>/
    "save_max": 8192,           # cap longest side of SAVED pngs (analysis is full-res)
    "alpha": 0.5,
    "roi_canvas_size": 768,     # interactive ROI drawing window size (px)
    "roi_brush_radius": 8,      # initial ROI brush radius (canvas px)
}


def load_pipeline_config(path):
    with open(path) as f:
        pp = json.load(f)
    for required in ("model", "config", "image"):
        if required not in pp:
            raise KeyError(f"pipeline config '{path}' is missing required key '{required}'")
    cfg = dict(DEFAULTS)
    cfg.update(pp)
    if cfg["pyramid"] is not None:
        z = float(cfg["pyramid"])
        if not (0.0 < z <= 1.0):
            raise ValueError(f"pyramid must be in (0, 1], got {z}")
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

    # training config (preprocess + tiling params)
    with open(pp["config"]) as f:
        cfg = json.load(f)
    p = Preprocessor(cfg)

    model, device, _ = load_model(pp["model"])

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

    # resolve pyramid zoom
    if pp["pyramid"] is not None:
        z = float(pp["pyramid"])
        print(f"pyramid zoom: {z}  (from config)")
    else:
        czi_max_px = cfg["preprocess"]["czi_max_px"]
        z = 1.0
        while z > 0.0 and (H * z) * (W * z) > czi_max_px:
            z *= 0.5
        print(f"pyramid zoom: {z}  (halving loop, czi_max_px={czi_max_px:.2e})")

    dt_gb = (H * z) * (W * z) * 4 / 1e9
    if dt_gb > 12:
        print(f"WARNING: distance transform at this zoom needs ~{dt_gb:.1f} GB RAM; "
              f"consider setting a smaller 'pyramid' zoom.")

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

    print(f"work dims: {H_w} x {W_w}   working microns/pixel: {work_um:.4f}")
    if not fg.any():
        print("WARNING: no foreground predicted above threshold — overlays will show no mask.")

    outdir = Path(pp["outdir"]) if pp["outdir"] else Path("saved/pipeline") / img_path.stem
    outdir.mkdir(parents=True, exist_ok=True)
    alpha = float(pp["alpha"])
    save_max = int(pp["save_max"])

    # 4. mask overlay (alpha 0.5)
    mask_overlay = blend(disp, (1.0, 0.0, 0.0), fg, alpha)
    save_overlay(mask_overlay, outdir / "mask_overlay.png", save_max)

    # 5. distance bands outward from foreground, colored with a cycling palette
    band_microns = float(pp["band_microns"])
    px_per_band = max(1, int(round(band_microns / work_um)))
    inv = (~fg).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)   # px to nearest foreground
    band = np.floor(dist / px_per_band).astype(np.int32)
    band[fg] = -1                                       # mark foreground
    n_bands = int(band.max()) + 1 if (~fg).any() else 0
    print(f"bands: {band_microns:.0f} um each = {px_per_band} px/band  ->  {n_bands} bands to edge")

    k = max(1, int(pp["palette_size"]))
    palette = cycling_palette(k)
    band_rgb = np.zeros_like(disp)
    bg = band >= 0
    band_rgb[bg] = palette[band[bg] % k]
    band_overlay = disp.copy()
    band_overlay[bg] = (1.0 - alpha) * disp[bg] + alpha * band_rgb[bg]
    band_overlay = blend(band_overlay, (1.0, 1.0, 1.0), fg, alpha)  # foreground distinct

    text = [
        f"{work_um:.3f} um/px   {band_microns:.0f}um = {px_per_band}px/band   {n_bands} bands",
        f"native {H}x{W}  zoom {z}",
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
