"""
Generate a self-contained HTML viewer for one slide: the RGB image with its
ground-truth mask as a white overlay whose opacity is adjustable via a slider
(plus wheel-zoom / drag-pan since these are gigapixel images downsampled for
browser display).

Usage:
    python quant/make_overlay_viewer.py --rgb <image.tiff> --mask <mask.tiff> \
        --stem AJKV056-10-1-of-2 --outdir quant/overlays/AJKV056-10-1of2 --max-dim 3000
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import tifffile

OVERLAY_COLOR = (255, 255, 255)  # white


def load_rgb_uint8(path):
    arr = tifffile.imread(str(path))
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.dtype == np.uint16:
        arr = np.round(arr.astype(np.float64) / 257.0).clip(0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        raise ValueError(f"unexpected RGB dtype {arr.dtype}")
    return arr


def load_mask_bool(path, mode='threshold'):
    """mode='threshold': single-channel-ish mask, foreground = value >= 128.
    mode='nonwhite': legacy Photoshop-exported layers where foreground = any
    channel != 255 (see quantify_hillestad.py's nonwhite_mask)."""
    arr = tifffile.imread(str(path))
    if mode == 'nonwhite':
        if arr.ndim == 2:
            return arr != 255
        m = arr[..., 0] != 255
        for c in range(1, arr.shape[2]):
            m |= arr[..., c] != 255
        return m
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr >= 128


def downsample(arr, max_dim, interp):
    h, w = arr.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1:
        return arr
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(arr, (new_w, new_h), interpolation=interp)


def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rgb = load_rgb_uint8(args.rgb)
    mask = load_mask_bool(args.mask, args.mask_mode)
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"rgb shape {rgb.shape[:2]} != mask shape {mask.shape}")

    rgb_small = downsample(rgb, args.max_dim, cv2.INTER_AREA)
    mask_small = downsample(mask.astype(np.uint8) * 255, args.max_dim, cv2.INTER_NEAREST) >= 128

    base_bgr = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(outdir / 'base.png'), base_bgr)

    h, w = mask_small.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[mask_small, 0] = OVERLAY_COLOR[2]
    overlay[mask_small, 1] = OVERLAY_COLOR[1]
    overlay[mask_small, 2] = OVERLAY_COLOR[0]
    overlay[mask_small, 3] = 255
    cv2.imwrite(str(outdir / 'mask_overlay.png'), overlay)

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{args.stem}: RGB + {args.mask_label} overlay</title>
<style>
  html, body {{ margin: 0; padding: 0; background: #1a1a1a; color: #eee;
                font-family: -apple-system, Helvetica, Arial, sans-serif; height: 100%; }}
  #bar {{ display: flex; align-items: center; gap: 12px; padding: 10px 16px;
          background: #222; border-bottom: 1px solid #333; }}
  #bar h1 {{ font-size: 14px; font-weight: 600; margin: 0; color: #fff; flex: 0 0 auto; }}
  #bar label {{ font-size: 13px; color: #bbb; }}
  #opacity, #offset-x, #offset-y {{ width: 160px; }}
  #opacity-val, #offset-x-val, #offset-y-val {{ font-size: 13px; color: #fff; width: 3.5em; }}
  #reset-offset {{ font-size: 12px; color: #ddd; background: #333; border: 1px solid #444;
                    border-radius: 4px; padding: 4px 8px; cursor: pointer; }}
  #reset-offset:hover {{ background: #3a3a3a; }}
  #hint {{ font-size: 12px; color: #888; margin-left: auto; }}
  #viewport {{ position: relative; width: 100%; height: calc(100% - 46px);
               overflow: hidden; cursor: grab; background: #000; }}
  #viewport.dragging {{ cursor: grabbing; }}
  #stage {{ position: absolute; top: 0; left: 0; transform-origin: 0 0; }}
  #stage img {{ display: block; position: absolute; top: 0; left: 0;
                width: {w}px; height: {h}px; user-select: none; pointer-events: none; }}
  #mask {{ opacity: 0.5; }}
</style>
</head>
<body>
<div id="bar">
  <h1>{args.stem}</h1>
  <label for="opacity">{args.mask_label} opacity</label>
  <input id="opacity" type="range" min="0" max="100" value="50">
  <span id="opacity-val">50%</span>
  <label for="offset-x">{args.mask_label} X offset</label>
  <input id="offset-x" type="range" min="-30" max="30" value="0">
  <span id="offset-x-val">0px</span>
  <label for="offset-y">{args.mask_label} Y offset</label>
  <input id="offset-y" type="range" min="-30" max="30" value="0">
  <span id="offset-y-val">0px</span>
  <button id="reset-offset">reset offset</button>
  <span id="hint">scroll to zoom, drag to pan, double-click to reset view</span>
</div>
<div id="viewport">
  <div id="stage">
    <img id="base" src="base.png">
    <img id="mask" src="mask_overlay.png">
  </div>
</div>
<script>
  const opacitySlider = document.getElementById('opacity');
  const opacityVal = document.getElementById('opacity-val');
  const maskImg = document.getElementById('mask');
  opacitySlider.addEventListener('input', () => {{
    maskImg.style.opacity = opacitySlider.value / 100;
    opacityVal.textContent = opacitySlider.value + '%';
  }});

  const offsetXSlider = document.getElementById('offset-x');
  const offsetYSlider = document.getElementById('offset-y');
  const offsetXVal = document.getElementById('offset-x-val');
  const offsetYVal = document.getElementById('offset-y-val');
  function applyMaskOffset() {{
    const dx = offsetXSlider.value, dy = offsetYSlider.value;
    maskImg.style.transform = `translate(${{dx}}px, ${{dy}}px)`;
    offsetXVal.textContent = dx + 'px';
    offsetYVal.textContent = dy + 'px';
  }}
  offsetXSlider.addEventListener('input', applyMaskOffset);
  offsetYSlider.addEventListener('input', applyMaskOffset);
  document.getElementById('reset-offset').addEventListener('click', () => {{
    offsetXSlider.value = 0; offsetYSlider.value = 0; applyMaskOffset();
  }});

  const viewport = document.getElementById('viewport');
  const stage = document.getElementById('stage');
  let scale = 1, tx = 0, ty = 0;
  let fitScale = 1;

  function fit() {{
    const vw = viewport.clientWidth, vh = viewport.clientHeight;
    fitScale = Math.min(vw / {w}, vh / {h});
    scale = fitScale;
    tx = (vw - {w} * scale) / 2;
    ty = (vh - {h} * scale) / 2;
    apply();
  }}

  function apply() {{
    stage.style.transform = `translate(${{tx}}px, ${{ty}}px) scale(${{scale}})`;
  }}

  viewport.addEventListener('wheel', (e) => {{
    e.preventDefault();
    const rect = viewport.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const factor = Math.pow(1.0015, -e.deltaY);
    const newScale = Math.min(Math.max(scale * factor, fitScale * 0.5), fitScale * 40);
    const ratio = newScale / scale;
    tx = mx - (mx - tx) * ratio;
    ty = my - (my - ty) * ratio;
    scale = newScale;
    apply();
  }}, {{ passive: false }});

  let dragging = false, lastX = 0, lastY = 0;
  viewport.addEventListener('mousedown', (e) => {{
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    viewport.classList.add('dragging');
  }});
  window.addEventListener('mouseup', () => {{ dragging = false; viewport.classList.remove('dragging'); }});
  window.addEventListener('mousemove', (e) => {{
    if (!dragging) return;
    tx += e.clientX - lastX; ty += e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    apply();
  }});
  viewport.addEventListener('dblclick', fit);
  window.addEventListener('resize', fit);
  fit();
</script>
</body>
</html>
"""
    (outdir / 'index.html').write_text(html)
    print(f"saved -> {outdir / 'index.html'}")
    print(f"saved -> {outdir / 'base.png'}")
    print(f"saved -> {outdir / 'mask_overlay.png'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rgb', required=True)
    parser.add_argument('--mask', required=True)
    parser.add_argument('--stem', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--max-dim', type=int, default=3000)
    parser.add_argument('--mask-mode', choices=['threshold', 'nonwhite'], default='threshold')
    parser.add_argument('--mask-label', default='GT mask')
    main(parser.parse_args())
