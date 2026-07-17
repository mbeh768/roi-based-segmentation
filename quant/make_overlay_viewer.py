"""
Generate a self-contained HTML viewer for one slide: the RGB image with its
ground-truth mask as a white overlay whose opacity and X/Y offset (in image
pixels, for checking mask-to-image registration) are adjustable via sliders
(plus wheel-zoom / drag-pan since these are gigapixel images downsampled for
browser display). Optionally also shows a second, independently toggleable
overlay: the mask region grown or shrunk (dilated/eroded) by an
offset-controlled number of pixels (up to +/-60px by default), for
visualizing margin/buffer zones around the mask. Growing colors the newly
added mass teal; shrinking fills the surviving region orange. Tracks
whichever mask (original or pockets-removed) is currently displayed.

Usage:
    python quant/make_overlay_viewer.py --rgb <image.tiff> --mask <mask.tiff> \
        --stem AJKV056-10-1-of-2 --outdir quant/overlays/AJKV056-10-1of2 --max-dim 3000
"""
import argparse
import base64
from pathlib import Path

import cv2
import numpy as np
import tifffile

OVERLAY_COLOR = (255, 255, 255)  # white
BAND_COLOR = (255, 140, 0)  # orange, for the eroded (shrunk) offset band
BAND_GROW_COLOR = (0, 191, 165)  # teal, for pixels newly added by a dilate (grow) offset
OTSU_COLOR = (255, 0, 255)  # magenta, for the Otsu-threshold foreground overlay
DIST_CLIP = 60  # px; keeps signed distance in [-60, 60], packed into 2 uint8 channels (+60)
POCKET_FRAC = 0.10  # remove connected foreground regions <= this fraction of the largest region's area
OTSU_ISLAND_FRAC = 0.05  # otsu "remove islands" cutoff, as a fraction of the largest otsu foreground blob's area


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


def signed_distance_encoded(mask, clip=DIST_CLIP):
    """Per-pixel distance (px) to the mask boundary, signed negative inside the
    mask and positive outside, clipped to +/-clip. A single uint8 channel only
    holds +/-127, which is too coarse for these gigapixel slides (a visible
    grow/shrink needs a range up to ~1000px), so the value is packed as a
    16-bit unsigned int (clip+value, range [0, 2*clip]) split across the R
    (high byte) and G (low byte) channels of a BGR image -- an 8-bit-per-
    channel PNG round-trips through <canvas> without precision loss, unlike a
    true 16-bit PNG (canvas always reads back 8-bit RGBA). Growing the mask by
    an offset o (dilate) then reduces to "keep pixels where value <= o";
    shrinking (erode) is the same test with a negative o."""
    fg = mask.astype(np.uint8)
    dist_out = cv2.distanceTransform(1 - fg, cv2.DIST_L2, 3)
    dist_in = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
    signed = np.where(mask, -dist_in, dist_out)
    clipped = np.clip(np.round(signed), -clip, clip).astype(np.int32)
    encoded = (clipped + clip).astype(np.uint16)
    high = (encoded >> 8).astype(np.uint8)
    low = (encoded & 0xFF).astype(np.uint8)
    h, w = mask.shape
    packed = np.zeros((h, w, 3), dtype=np.uint8)
    packed[..., 2] = high  # R
    packed[..., 1] = low   # G
    return packed


def remove_pockets(mask, frac=POCKET_FRAC):
    """Drop connected foreground components whose area is <= frac * the area of
    the largest component (removes small isolated pockets, keeps the main
    region and any other comparably large regions)."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask.copy()
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = areas.max()
    keep = {i + 1 for i, a in enumerate(areas) if a > frac * largest}
    return np.isin(labels, list(keep))


def otsu_foreground(base_bgr):
    """Global Otsu threshold on the red channel (this project's fluorescence
    RGB exports carry the strongest tissue/background separation on R --
    see quantify_intensity.py's R/G/B band-intensity split), foreground =
    brighter class (fluorescent tissue), background = darker class (the
    near-black background these images are shot against -- confirmed
    empirically: background corner pixels read ~0, tissue reads tens-to-
    hundreds). Used as a cheap validity check for whether a single global
    intensity threshold can locate the epicardial boundary (tissue/background
    edge) as a stand-in for the annotated mask."""
    red = base_bgr[..., 2]
    _, fg = cv2.threshold(red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return fg.astype(bool)


def fill_enclosed_holes(mask, frac=POCKET_FRAC):
    """Fill background components fully enclosed by foreground (i.e. not
    touching the image border) whose area is <= frac * the area of the
    largest foreground component -- 'pockets' of background surrounded by
    tissue, as opposed to islands of isolated foreground (see
    remove_pockets, which does the reverse: dropping small foreground
    blobs)."""
    fg_num_labels, _, fg_stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if fg_num_labels <= 1:
        return mask.copy()
    largest_fg = fg_stats[1:, cv2.CC_STAT_AREA].max()

    h, w = mask.shape
    bg_num_labels, bg_labels, bg_stats, _ = cv2.connectedComponentsWithStats(
        (~mask).astype(np.uint8), connectivity=8)
    result = mask.copy()
    for i in range(1, bg_num_labels):
        x, y, bw, bh, area = bg_stats[i, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                           cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT,
                                           cv2.CC_STAT_AREA]]
        touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if touches_border:
            continue
        if area <= frac * largest_fg:
            result[bg_labels == i] = True
    return result


def downsample(arr, max_dim, interp):
    h, w = arr.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1:
        return arr
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(arr, (new_w, new_h), interpolation=interp)


def render_viewer(outdir, base_bgr, mask_small, stem, mask_label, band_max_offset=60):
    """Write base.png, mask_overlay.png, mask_overlay_clean.png and index.html
    into outdir, given an already-downsampled BGR base image and boolean mask
    at matching (h, w). This is the shared core used both by the CLI (after
    downsampling a fresh RGB/mask pair) and by a regeneration script that
    reuses a directory's existing base.png/mask_overlay.png.

    The grow/shrink distance map is NOT written as a separate PNG: these
    viewers are meant to be opened directly (file://, no local server), and
    Chrome taints a canvas that has drawn an image loaded via a relative
    file:// src, which silently breaks getImageData() for the band feature.
    Embedding the same PNG bytes as a base64 data: URI instead avoids the
    taint (data: URIs don't trigger the cross-origin canvas restriction)
    while keeping the same PNG compression -- confirmed empirically since
    this is a real Chromium quirk, not documented behavior to take on faith."""
    if band_max_offset > DIST_CLIP:
        raise ValueError(f"band_max_offset ({band_max_offset}) exceeds DIST_CLIP ({DIST_CLIP})")
    if base_bgr.shape[:2] != mask_small.shape:
        raise ValueError(f"base shape {base_bgr.shape[:2]} != mask shape {mask_small.shape}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(outdir / 'base.png'), base_bgr)

    h, w = mask_small.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[mask_small, 0] = OVERLAY_COLOR[2]
    overlay[mask_small, 1] = OVERLAY_COLOR[1]
    overlay[mask_small, 2] = OVERLAY_COLOR[0]
    overlay[mask_small, 3] = 255
    cv2.imwrite(str(outdir / 'mask_overlay.png'), overlay)

    mask_clean = remove_pockets(mask_small, frac=POCKET_FRAC)
    overlay_clean = np.zeros((h, w, 4), dtype=np.uint8)
    overlay_clean[mask_clean, 0] = OVERLAY_COLOR[2]
    overlay_clean[mask_clean, 1] = OVERLAY_COLOR[1]
    overlay_clean[mask_clean, 2] = OVERLAY_COLOR[0]
    overlay_clean[mask_clean, 3] = 255
    cv2.imwrite(str(outdir / 'mask_overlay_clean.png'), overlay_clean)

    def write_otsu_overlay(m, filename):
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[m, 0] = OTSU_COLOR[2]
        overlay[m, 1] = OTSU_COLOR[1]
        overlay[m, 2] = OTSU_COLOR[0]
        overlay[m, 3] = 255
        cv2.imwrite(str(outdir / filename), overlay)

    otsu_mask = otsu_foreground(base_bgr)
    otsu_mask_noislands = remove_pockets(otsu_mask, frac=OTSU_ISLAND_FRAC)
    otsu_mask_nopockets = fill_enclosed_holes(otsu_mask, frac=POCKET_FRAC)
    otsu_mask_clean = fill_enclosed_holes(otsu_mask_noislands, frac=POCKET_FRAC)
    write_otsu_overlay(otsu_mask, 'otsu_overlay.png')
    write_otsu_overlay(otsu_mask_noislands, 'otsu_overlay_noislands.png')
    write_otsu_overlay(otsu_mask_nopockets, 'otsu_overlay_nopockets.png')
    write_otsu_overlay(otsu_mask_clean, 'otsu_overlay_clean.png')

    def encode_dist_b64(m):
        dist_map = signed_distance_encoded(m, clip=DIST_CLIP)
        ok, dist_png_bytes = cv2.imencode('.png', dist_map)
        if not ok:
            raise RuntimeError("failed to encode distance map PNG")
        return base64.b64encode(dist_png_bytes.tobytes()).decode('ascii')

    # Two separate distance maps: the grow/shrink band must reflect whichever
    # mask (original or pockets-removed) is currently displayed -- a pocket
    # dropped from view but still baked into the distance transform would
    # keep distorting the band around where it used to be.
    dist_b64_orig = encode_dist_b64(mask_small)
    dist_b64_clean = encode_dist_b64(mask_clean)

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{stem}: RGB + {mask_label} overlay</title>
<style>
  html, body {{ margin: 0; padding: 0; background: #1a1a1a; color: #eee;
                font-family: -apple-system, Helvetica, Arial, sans-serif; height: 100%; }}
  #bar {{ display: flex; align-items: center; gap: 12px; padding: 10px 16px;
          background: #222; border-bottom: 1px solid #333; }}
  #bar h1 {{ font-size: 14px; font-weight: 600; margin: 0; color: #fff; flex: 0 0 auto; }}
  #bar label {{ font-size: 13px; color: #bbb; }}
  #opacity, #offset-x, #offset-y, #band-offset {{ width: 160px; }}
  #opacity-val, #offset-x-val, #offset-y-val, #band-offset-val {{ font-size: 13px; color: #fff; width: 3.5em; }}
  #reset-offset, #pockets-toggle, #otsu-islands-toggle, #otsu-pockets-toggle {{
                    font-size: 12px; color: #ddd; background: #333; border: 1px solid #444;
                    border-radius: 4px; padding: 4px 8px; cursor: pointer; }}
  #reset-offset:hover, #pockets-toggle:hover, #otsu-islands-toggle:hover, #otsu-pockets-toggle:hover {{ background: #3a3a3a; }}
  #pockets-toggle.active, #otsu-islands-toggle.active, #otsu-pockets-toggle.active {{
                    background: #2d6a3e; border-color: #3a8a51; color: #fff; }}
  #pockets-toggle.active:hover, #otsu-islands-toggle.active:hover, #otsu-pockets-toggle.active:hover {{ background: #357a48; }}
  #bar-row2 {{ display: flex; align-items: center; gap: 12px; padding: 8px 16px;
               background: #1e1e1e; border-bottom: 1px solid #333; }}
  #hint {{ font-size: 12px; color: #888; margin-left: auto; }}
  #viewport {{ position: relative; width: 100%; height: calc(100% - 84px);
               overflow: hidden; cursor: grab; background: #000; }}
  #viewport.dragging {{ cursor: grabbing; }}
  #stage {{ position: absolute; top: 0; left: 0; transform-origin: 0 0; }}
  #stage img, #stage canvas {{ display: block; position: absolute; top: 0; left: 0;
                width: {w}px; height: {h}px; user-select: none; pointer-events: none; }}
  #mask {{ opacity: 0.5; }}
  #band {{ display: none; opacity: 0.55; }}
  #otsu {{ opacity: 0; }}
  #bar-row3 {{ display: flex; align-items: center; gap: 12px; padding: 8px 16px;
               background: #1e1e1e; border-bottom: 1px solid #333; }}
</style>
</head>
<body>
<div id="bar">
  <h1>{stem}</h1>
  <label for="opacity">{mask_label} opacity</label>
  <input id="opacity" type="range" min="0" max="100" value="50">
  <span id="opacity-val">50%</span>
  <label for="offset-x">{mask_label} X offset</label>
  <input id="offset-x" type="range" min="-30" max="30" value="0">
  <span id="offset-x-val">0px</span>
  <label for="offset-y">{mask_label} Y offset</label>
  <input id="offset-y" type="range" min="-30" max="30" value="0">
  <span id="offset-y-val">0px</span>
  <button id="reset-offset">reset offset</button>
  <button id="pockets-toggle" title="hide connected foreground regions &lt;= {int(POCKET_FRAC * 100)}% the size of the largest region">remove pockets (&lt;={int(POCKET_FRAC * 100)}%)</button>
  <span id="hint">scroll to zoom, drag to pan, double-click to reset view</span>
</div>
<div id="bar-row2">
  <label><input type="checkbox" id="band-toggle"> show {mask_label} grow/shrink band</label>
  <label for="band-offset">offset</label>
  <input id="band-offset" type="range" min="-{band_max_offset}" max="{band_max_offset}" value="0" disabled>
  <span id="band-offset-val">0px</span>
  <span id="band-hint" style="font-size:12px; color:#888;">positive = grow (dilate): teal = added mass; negative = shrink (erode): orange = surviving region</span>
</div>
<div id="bar-row3">
  <label><input type="checkbox" id="otsu-toggle"> show Otsu threshold foreground</label>
  <label for="otsu-opacity">opacity</label>
  <input id="otsu-opacity" type="range" min="0" max="100" value="50">
  <span id="otsu-opacity-val">50%</span>
  <button id="otsu-islands-toggle" title="drop isolated foreground blobs &lt;= {int(OTSU_ISLAND_FRAC * 100)}% the size of the largest blob">remove islands (&lt;={int(OTSU_ISLAND_FRAC * 100)}%)</button>
  <button id="otsu-pockets-toggle" title="fill enclosed background holes &lt;= {int(POCKET_FRAC * 100)}% the size of the largest foreground blob">remove pockets (&lt;={int(POCKET_FRAC * 100)}%)</button>
  <span style="font-size:12px; color:#888;">magenta = pixels classified foreground (tissue) by a global Otsu threshold on the red channel -- validity check against the {mask_label} boundary as an epicardium detector</span>
</div>
<div id="viewport">
  <div id="stage">
    <img id="base" src="base.png">
    <img id="mask" src="mask_overlay.png">
    <img id="otsu" src="otsu_overlay.png">
    <canvas id="band" width="{w}" height="{h}"></canvas>
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

  const pocketsToggle = document.getElementById('pockets-toggle');
  let pocketsRemoved = false;
  pocketsToggle.addEventListener('click', async () => {{
    pocketsRemoved = !pocketsRemoved;
    maskImg.src = pocketsRemoved ? 'mask_overlay_clean.png' : 'mask_overlay.png';
    pocketsToggle.classList.toggle('active', pocketsRemoved);
    if (bandToggle.checked) {{
      await loadDistMap(pocketsRemoved);
      scheduleRenderBand();
    }}
  }});

  const otsuToggle = document.getElementById('otsu-toggle');
  const otsuImg = document.getElementById('otsu');
  const otsuOpacitySlider = document.getElementById('otsu-opacity');
  const otsuOpacityVal = document.getElementById('otsu-opacity-val');
  const otsuIslandsToggle = document.getElementById('otsu-islands-toggle');
  const otsuPocketsToggle = document.getElementById('otsu-pockets-toggle');
  let otsuNoIslands = false, otsuNoPockets = false;
  function otsuSrc() {{
    if (otsuNoIslands && otsuNoPockets) return 'otsu_overlay_clean.png';
    if (otsuNoIslands) return 'otsu_overlay_noislands.png';
    if (otsuNoPockets) return 'otsu_overlay_nopockets.png';
    return 'otsu_overlay.png';
  }}
  function applyOtsuOpacity() {{
    otsuImg.style.opacity = otsuToggle.checked ? otsuOpacitySlider.value / 100 : 0;
  }}
  otsuToggle.addEventListener('change', applyOtsuOpacity);
  otsuOpacitySlider.addEventListener('input', () => {{
    otsuOpacityVal.textContent = otsuOpacitySlider.value + '%';
    applyOtsuOpacity();
  }});
  otsuIslandsToggle.addEventListener('click', () => {{
    otsuNoIslands = !otsuNoIslands;
    otsuIslandsToggle.classList.toggle('active', otsuNoIslands);
    otsuImg.src = otsuSrc();
  }});
  otsuPocketsToggle.addEventListener('click', () => {{
    otsuNoPockets = !otsuNoPockets;
    otsuPocketsToggle.classList.toggle('active', otsuNoPockets);
    otsuImg.src = otsuSrc();
  }});

  const bandToggle = document.getElementById('band-toggle');
  const bandOffsetSlider = document.getElementById('band-offset');
  const bandOffsetVal = document.getElementById('band-offset-val');
  const bandCanvas = document.getElementById('band');
  const bandCtx = bandCanvas.getContext('2d');
  const BAND_COLOR = [{BAND_COLOR[0]}, {BAND_COLOR[1]}, {BAND_COLOR[2]}];       // erode: shrunk region
  const BAND_GROW_COLOR = [{BAND_GROW_COLOR[0]}, {BAND_GROW_COLOR[1]}, {BAND_GROW_COLOR[2]}]; // dilate: newly-added mass
  const DIST_CLIP = {DIST_CLIP};
  // Uint8ClampedArray RGBA caches; R=high byte, G=low byte of (signed_dist_px + DIST_CLIP).
  // Separate caches for the original vs. pockets-removed mask -- the band must
  // reflect whichever one is currently displayed.
  let distDataOrig = null;
  let distDataClean = null;
  let bandPending = false;

  // Embedded (not loaded from a separate file) so opening this HTML directly via
  // file:// still works: Chrome taints the canvas for an image loaded via a
  // relative file:// src, which silently breaks getImageData() below. A data:
  // URI of the same PNG bytes does not taint the canvas.
  const DIST_PNG_DATA_URI_ORIG = 'data:image/png;base64,{dist_b64_orig}';
  const DIST_PNG_DATA_URI_CLEAN = 'data:image/png;base64,{dist_b64_clean}';

  function loadDistMap(useClean) {{
    if (useClean && distDataClean) return Promise.resolve();
    if (!useClean && distDataOrig) return Promise.resolve();
    return new Promise((resolve) => {{
      const img = new Image();
      img.onload = () => {{
        const off = document.createElement('canvas');
        off.width = {w}; off.height = {h};
        const offCtx = off.getContext('2d');
        offCtx.drawImage(img, 0, 0);
        const data = offCtx.getImageData(0, 0, {w}, {h}).data;
        if (useClean) distDataClean = data; else distDataOrig = data;
        resolve();
      }};
      img.src = useClean ? DIST_PNG_DATA_URI_CLEAN : DIST_PNG_DATA_URI_ORIG;
    }});
  }}

  function renderBand() {{
    bandPending = false;
    const distData = pocketsRemoved ? distDataClean : distDataOrig;
    if (!distData) return;
    const offset = parseInt(bandOffsetSlider.value, 10);
    const out = bandCtx.createImageData({w}, {h});
    const dst = out.data;
    const [er, eg, eb] = BAND_COLOR;
    const [gr, gg, gb] = BAND_GROW_COLOR;
    for (let i = 0; i < distData.length; i += 4) {{
      const signed = (distData[i] << 8 | distData[i + 1]) - DIST_CLIP;
      if (offset <= 0) {{
        // erode: fill the surviving (shrunk) region so its smaller boundary
        // is visible against the full-size white mask underneath.
        if (signed <= offset) {{
          dst[i] = er; dst[i + 1] = eg; dst[i + 2] = eb; dst[i + 3] = 255;
        }}
      }} else {{
        // dilate: only color pixels that are newly added (were background,
        // now within the grow offset) -- the original mask stays uncolored
        // so the teal ring shows exactly what growth adds.
        if (signed > 0 && signed <= offset) {{
          dst[i] = gr; dst[i + 1] = gg; dst[i + 2] = gb; dst[i + 3] = 255;
        }}
      }}
    }}
    bandCtx.putImageData(out, 0, 0);
  }}

  function scheduleRenderBand() {{
    if (bandPending) return;
    bandPending = true;
    requestAnimationFrame(renderBand);
  }}

  bandToggle.addEventListener('change', async () => {{
    bandOffsetSlider.disabled = !bandToggle.checked;
    if (bandToggle.checked) {{
      bandCanvas.style.display = 'block';
      await loadDistMap(pocketsRemoved);
      scheduleRenderBand();
    }} else {{
      bandCanvas.style.display = 'none';
    }}
  }});
  bandOffsetSlider.addEventListener('input', () => {{
    bandOffsetVal.textContent = bandOffsetSlider.value + 'px';
    if (bandToggle.checked) scheduleRenderBand();
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
    print(f"saved -> {outdir / 'mask_overlay_clean.png'}")
    print(f"saved -> {outdir / 'otsu_overlay.png'}")
    print(f"saved -> {outdir / 'otsu_overlay_noislands.png'}")
    print(f"saved -> {outdir / 'otsu_overlay_nopockets.png'}")
    print(f"saved -> {outdir / 'otsu_overlay_clean.png'}")


def main(args):
    rgb = load_rgb_uint8(args.rgb)
    mask = load_mask_bool(args.mask, args.mask_mode)
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"rgb shape {rgb.shape[:2]} != mask shape {mask.shape}")

    rgb_small = downsample(rgb, args.max_dim, cv2.INTER_AREA)
    mask_small = downsample(mask.astype(np.uint8) * 255, args.max_dim, cv2.INTER_NEAREST) >= 128
    base_bgr = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR)

    render_viewer(args.outdir, base_bgr, mask_small, args.stem, args.mask_label,
                  band_max_offset=args.band_max_offset)


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
    parser.add_argument('--band-max-offset', type=int, default=60,
                         help='max |px| for the grow/shrink band slider (<= %d)' % DIST_CLIP)
    main(parser.parse_args())
