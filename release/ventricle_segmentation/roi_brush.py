"""
Interactive ROI brush for a single slide.

The segmentation model was trained with `use_roi=True`, meaning every tile it
ever saw was paired with a hand-drawn region-of-interest mask marking roughly
where the ventricle wall sits. It therefore needs one at inference time too --
there is no "just run it on the image" path that matches how it was trained.

This module is a single-image adaptation of the annotation brush in the research
repo's annotate_globals.py (BrushROIDrawer + the hover-tooltip keybind panel).
Unlike the stripped-down copy in demo.py, the full keybind list is shown in its
own window and hovering a row prints the long explanation, so a collaborator who
has never seen the tool can discover the controls without reading the README.

The ROI is drawn on the model's own downsampled "global" view of the slide,
which is the exact resolution the model's ROI input branch expects.
"""
import time

import cv2
import numpy as np
import torch


# (key, short label, long tooltip shown on hover)
KEYBINDS = [
    ('L-click / drag', 'paint ROI',
     'Hold the left mouse button and drag to paint the ROI in orange. Paint a '
     'band that covers the ventricle wall you want segmented.'),
    ('Shift + L-drag', 'erase ROI',
     'Hold Shift and left-click / drag to erase. Use this on a filled mask to '
     'carve out islands or holes you painted by accident.'),
    ('- / =', 'shrink / grow brush',
     'Adjust the brush radius. Step size scales with the current radius, so it '
     'stays usable at both very small and very large sizes. The slider at the '
     'top of this window does the same thing.'),
    ('f', 'flood-fill',
     'Fill enclosed regions inside the outline you drew, so you can trace a '
     'ring instead of scribbling it solid. Shows a side-by-side preview first: '
     'press y to accept, n to reject.'),
    ('y / n', 'confirm fill',
     'Only active during the flood-fill preview: y commits the filled mask, '
     'n or ESC discards the fill and keeps your original outline.'),
    ('[ / ]', 'darken / brighten',
     'Adjust display gamma only. This changes how the slide looks on screen to '
     'help you see faint tissue; it does NOT change the image fed to the model.'),
    ('\\', 'reset brightness',
     'Reset display gamma back to 1.0 (the unadjusted view).'),
    ('c', 'clear',
     'Erase everything painted on this slide and start over.'),
    ('s / Enter', 'accept + segment',
     'Accept the ROI you drew and start segmentation. This is the button you '
     'press when you are done drawing.'),
    ('q / ESC', 'cancel',
     'Close without segmenting. Nothing is written to disk.'),
]

BRUSH_MAX = 80


def _wrap_text(text, max_chars=46):
    words = text.split()
    lines, cur = [], ''
    for w in words:
        if len(cur) + 1 + len(w) > max_chars and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + ' ' + w).strip()
    if cur:
        lines.append(cur)
    return lines


def _render_keybinds(hover_row, cursor_xy):
    """Draw the keybind reference panel, with a tooltip for the hovered row."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    pad = 10
    row_h = 24
    key_col_w = 170
    width = 560
    header_h = 30
    height = pad * 2 + header_h + row_h * len(KEYBINDS)
    img = np.full((height, width, 3), 30, dtype=np.uint8)

    cv2.putText(img, 'CONTROLS  (hover a row for details)', (pad, pad + 18),
                font, 0.55, (120, 220, 160), 1, cv2.LINE_AA)

    for idx, (key, short, _) in enumerate(KEYBINDS):
        y0 = pad + header_h + idx * row_h
        if idx == hover_row:
            cv2.rectangle(img, (0, y0), (width, y0 + row_h), (60, 60, 90), -1)
        baseline = y0 + row_h - 7
        cv2.putText(img, key, (pad, baseline), font, scale, (180, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(img, short, (pad + key_col_w, baseline), font, scale,
                    (220, 220, 220), 1, cv2.LINE_AA)

    if hover_row is not None and cursor_xy is not None:
        lines = _wrap_text(KEYBINDS[hover_row][2])
        tw = max(cv2.getTextSize(ln, font, scale, 1)[0][0] for ln in lines) + 12
        th = len(lines) * 18 + 10
        cx, cy = cursor_xy
        x = min(width - tw - 4, max(4, cx + 12))
        y = min(height - th - 4, max(4, cy + 12))
        cv2.rectangle(img, (x, y), (x + tw, y + th), (0, 0, 0), -1)
        cv2.rectangle(img, (x, y), (x + tw, y + th), (120, 120, 160), 1)
        for li, ln in enumerate(lines):
            cv2.putText(img, ln, (x + 6, y + 16 + li * 18), font, scale,
                        (230, 230, 230), 1, cv2.LINE_AA)
    return img


def _hover_row_at(y):
    """Map a y pixel in the keybind window back to a KEYBINDS index."""
    pad, header_h, row_h = 10, 30, 24
    idx = (y - pad - header_h) // row_h
    return int(idx) if 0 <= idx < len(KEYBINDS) else None


def _draw_banner(img, lines):
    if isinstance(lines, str):
        lines = [lines]
    scale = 0.55
    pad = 6
    sizes = [cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0] for ln in lines]
    box_w = max(w for w, _ in sizes) + pad * 2
    line_h = max(h for _, h in sizes) + 6
    box_h = line_h * len(lines) + pad
    cv2.rectangle(img, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    for i, ln in enumerate(lines):
        y = pad + (i + 1) * line_h - 6
        cv2.putText(img, ln, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 255, 0), 2)


def _fill_holes(mask):
    """Fill enclosed regions inside a binary outline mask (uint8 {0,1}).

    Note the limitation: only regions the background flood cannot reach are
    filled. A stroke with even a one-pixel gap, or one whose interior opens onto
    the edge of the canvas, is not enclosed and nothing happens. The caller
    reports that case rather than failing silently.
    """
    pad = cv2.copyMakeBorder(mask.astype(np.uint8), 1, 1, 1, 1,
                             cv2.BORDER_CONSTANT, value=0)
    flood = pad.copy()
    cv2.floodFill(flood, None, (0, 0), 2)
    holes = (flood == 0).astype(np.uint8)
    return (pad | holes)[1:-1, 1:-1]


def _apply_gamma(img_bgr, gamma):
    """Return a gamma-adjusted uint8 copy. gamma > 1 brightens, < 1 darkens."""
    if abs(gamma - 1.0) < 1e-3:
        return img_bgr
    inv = 1.0 / max(gamma, 1e-6)
    lut = (np.linspace(0, 1, 256) ** inv * 255).astype(np.uint8)
    return cv2.LUT(img_bgr, lut)


def global_to_bgr(tensor):
    """(C, H, W) float tensor -> contrast-stretched uint8 BGR for display."""
    t = tensor.detach().cpu().float().numpy()
    lo, hi = t.min(), t.max()
    t = (t - lo) / (hi - lo + 1e-8)
    arr = (t * 255).astype(np.uint8)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        arr = arr.transpose(1, 2, 0)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


class BrushROIDrawer:
    """Freeform circular brush. The mask lives at canvas resolution."""

    def __init__(self, radius=8, fill_alpha=0.45, fill_color=(0, 180, 255),
                 cursor_color=(0, 255, 255)):
        self.radius = max(1, int(radius))
        self.fill_alpha = fill_alpha
        self.fill_color = np.array(fill_color, dtype=np.float32)
        self.cursor_color = cursor_color
        self.canvas_h = 0
        self.canvas_w = 0
        self.mask = np.zeros((1, 1), dtype=np.uint8)
        self._mouse_down = None
        self._last_xy = None
        self._cursor = None

    def reset(self, canvas_h, canvas_w):
        self.canvas_h = canvas_h
        self.canvas_w = canvas_w
        self.mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        self._mouse_down = None
        self._last_xy = None
        self._cursor = None

    def clear(self):
        self.mask[:] = 0

    def set_radius(self, radius):
        self.radius = max(1, int(radius))

    def _stamp(self, x, y, value):
        cv2.circle(self.mask, (int(x), int(y)), self.radius, int(value), -1,
                   lineType=cv2.LINE_8)

    def _stroke(self, x, y, value):
        if self._last_xy is not None:
            cv2.line(self.mask, self._last_xy, (int(x), int(y)), int(value),
                     thickness=self.radius * 2, lineType=cv2.LINE_8)
        self._stamp(x, y, value)
        self._last_xy = (int(x), int(y))

    def on_mouse(self, event, x, y, flags):
        self._cursor = (int(x), int(y))
        # Erase = Shift + L-click. Shift state is read live, so toggling Shift
        # mid-drag switches the stroke between paint and erase.
        erase = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        value = 0 if erase else 1
        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_down = 'L'
            self._last_xy = None
            self._stroke(x, y, value)
        elif event == cv2.EVENT_LBUTTONUP:
            self._mouse_down = None
            self._last_xy = None
        elif event == cv2.EVENT_MOUSEMOVE and self._mouse_down == 'L':
            self._stroke(x, y, value)

    def render_overlay(self, canvas_bgr):
        h, w = canvas_bgr.shape[:2]
        out = canvas_bgr.astype(np.float32)
        filled = self.mask.astype(bool)
        fill_layer = np.tile(self.fill_color, (h, w, 1))
        alpha = np.where(filled[..., None], self.fill_alpha, 0.0).astype(np.float32)
        out = out * (1 - alpha) + fill_layer * alpha
        out = np.clip(out, 0, 255).astype(np.uint8)
        if self._cursor is not None:
            cv2.circle(out, self._cursor, self.radius, self.cursor_color, 1,
                       lineType=cv2.LINE_AA)
        return out

    def get_mask(self, target_h, target_w):
        full = cv2.resize(self.mask, (target_w, target_h),
                          interpolation=cv2.INTER_NEAREST).astype(np.float32)
        return torch.from_numpy(full).unsqueeze(0)


def draw_roi(global_tensor, slide_name='', canvas_size=768, brush_radius=8):
    """Open the brush UI over the model's global view of a slide.

    Args:
        global_tensor: (C, H, W) float tensor -- the model's downsampled view.
        slide_name: shown in the window banner.
        canvas_size: on-screen drawing resolution (square).
        brush_radius: starting brush radius in canvas pixels.

    Returns:
        (1, H, W) float32 ROI tensor at the global tensor's own resolution,
        or None if the user cancelled.
    """
    _, H, W = global_tensor.shape
    bgr = global_to_bgr(global_tensor)
    canvas = cv2.resize(bgr, (canvas_size, canvas_size), interpolation=cv2.INTER_LINEAR)

    drawer = BrushROIDrawer(radius=brush_radius)
    drawer.reset(canvas_size, canvas_size)

    win = 'draw ROI  -  paint over the ventricle wall'
    keys_win = 'controls'
    keys_state = {'hover': None, 'cursor': None}

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, canvas_size, canvas_size)
    cv2.setMouseCallback(win, lambda e, x, y, f, _u: drawer.on_mouse(e, x, y, f))

    def _on_keys_mouse(event, x, y, flags, _userdata):
        if event == cv2.EVENT_MOUSEMOVE:
            keys_state['hover'] = _hover_row_at(y)
            keys_state['cursor'] = (int(x), int(y))
        elif event == cv2.EVENT_MOUSELEAVE:
            keys_state['hover'] = None
            keys_state['cursor'] = None

    panel_h = 20 + 30 + 24 * len(KEYBINDS) + 40
    cv2.namedWindow(keys_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(keys_win, 560, panel_h)
    cv2.setMouseCallback(keys_win, _on_keys_mouse)
    cv2.createTrackbar('brush', keys_win, min(BRUSH_MAX, brush_radius), BRUSH_MAX,
                       lambda v: drawer.set_radius(max(1, v)))

    accepted = False
    gamma = 1.0
    pending_fill = None
    notice = ''
    notice_until = 0.0
    while True:
        gamma_canvas = _apply_gamma(canvas, gamma)

        if pending_fill is not None:
            left = gamma_canvas.copy()
            saved_mask = drawer.mask
            drawer.mask = pending_fill
            right = drawer.render_overlay(gamma_canvas)
            drawer.mask = saved_mask
            view = np.hstack([left, right])
            _draw_banner(view, ['FILL PREVIEW   (before | after)',
                                'y = accept fill    n / ESC = reject'])
            cv2.imshow(win, view)
            cv2.imshow(keys_win, _render_keybinds(keys_state['hover'], keys_state['cursor']))
            key = cv2.waitKey(15) & 0xFF
            if key == ord('y'):
                drawer.mask = pending_fill
                pending_fill = None
            elif key in (ord('n'), 27):
                pending_fill = None
            continue

        view = drawer.render_overlay(gamma_canvas)
        painted = int(drawer.mask.sum())
        banner = [
            f'{slide_name}   brush={drawer.radius}px   gamma={gamma:.2f}',
            's / Enter = accept + segment      q / ESC = cancel'
            + ('' if painted else '      (nothing painted yet)'),
        ]
        if notice and time.time() < notice_until:
            banner.append(notice)
        _draw_banner(view, banner)
        cv2.imshow(win, view)
        cv2.imshow(keys_win, _render_keybinds(keys_state['hover'], keys_state['cursor']))
        key = cv2.waitKey(15) & 0xFF

        if key in (ord('s'), 13, 10):          # s, Enter (LF/CR)
            if painted == 0:
                # Refuse to accept an empty ROI: the model would see no valid
                # tiles at all and return an all-zero mask, which looks like a
                # silent failure rather than user error.
                continue
            accepted = True
            break
        if key in (ord('q'), 27):
            break
        if key == ord('c'):
            drawer.clear()
        elif key == ord('f'):
            candidate = _fill_holes(drawer.mask)
            added = int(candidate.sum()) - int(drawer.mask.sum())
            if added > 0:
                pending_fill = candidate
                print(f"  fill: {added:,} px would be added "
                      f"-- press y to accept, n to reject", flush=True)
            else:
                # Nothing was enclosed. Say so instead of showing a preview
                # identical to the current mask, which just looks broken.
                pending_fill = None
                notice = ('nothing to fill: the outline is not closed, '
                          'or its inside touches the edge of the image')
                notice_until = time.time() + 3.0
                print(f"  fill: {notice}", flush=True)
        elif key in (ord('['), ord('{')):
            gamma = max(0.1, round(gamma - 0.25, 2))
        elif key in (ord(']'), ord('}')):
            gamma = min(8.0, round(gamma + 0.25, 2))
        elif key == ord('\\'):
            gamma = 1.0
        elif key in (ord('-'), ord('_')):
            drawer.set_radius(drawer.radius - max(1, drawer.radius // 8))
            cv2.setTrackbarPos('brush', keys_win, min(BRUSH_MAX, drawer.radius))
        elif key in (ord('='), ord('+')):
            drawer.set_radius(drawer.radius + max(1, drawer.radius // 8))
            cv2.setTrackbarPos('brush', keys_win, min(BRUSH_MAX, drawer.radius))

    cv2.destroyWindow(win)
    cv2.destroyWindow(keys_win)
    cv2.waitKey(1)
    if not accepted:
        return None
    return drawer.get_mask(H, W)
