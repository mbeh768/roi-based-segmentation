"""
Rapid-fire ROI annotation for every global.pt in a preprocessed data dir.

Each slide gets one or more ROIs written to:
    <data_dir>/images/<stem>/roi_<stem>.pt        shape (1, H, W) float32 in {0,1}
    <data_dir>/images/<stem>/roi_<stem>_1.pt      additional ROI for same slide
    <data_dir>/images/<stem>/roi_<stem>_2.pt      ...

UI is swappable via the `ROIDrawer` interface. The default brush mode uses
`BrushROIDrawer` (freeform circular brush). With --showgt, `GtTileROIDrawer`
is used instead: the ground-truth mask is displayed under an 8x8 tile grid and
each click copies GT foreground from the clicked tile into the ROI.

Usage:
    python annotate_globals.py --data-dir data/[collapsed]64_tiles_per_image
    python annotate_globals.py --data-dir ... --brush-radius 24 --canvas-size 768
    python annotate_globals.py --data-dir ... --overwrite
    python annotate_globals.py --data-dir ... --start-from <stem>
    python annotate_globals.py --data-dir ... --add                       # add additional ROIs to slides that already have one
    python annotate_globals.py --data-dir ... --showgt --add              # build ROIs from the GT mask by clicking tiles

Keys:
    L-click / drag     paint ROI
    R-click / drag     erase ROI
    -  /  =            shrink / grow brush
    f                  flood-fill enclosed regions, then preview (y accept / n reject)
    s                  save current ROI and advance
    c                  clear current ROI
    b                  go back one slide
    [  /  ]            decrease / increase preview gamma (brighten / darken)
    \\                 reset gamma to 1.0
    q / ESC            quit
"""
import argparse
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import torch


class ROIDrawer(ABC):
    """UI-mode interface. Swap implementations (grid, brush, polygon) freely."""

    @abstractmethod
    def reset(self, canvas_h, canvas_w):
        ...

    @abstractmethod
    def clear(self):
        ...

    @abstractmethod
    def on_mouse(self, event, x, y, flags):
        ...

    @abstractmethod
    def render_overlay(self, canvas_bgr):
        ...

    @abstractmethod
    def get_mask(self, target_h, target_w):
        """Return final ROI mask as (1, target_h, target_w) float32 torch tensor."""
        ...

    @abstractmethod
    def load_existing(self, mask_np):
        """Initialize state from an existing (H, W) {0,1} float32 array."""
        ...


class BrushROIDrawer(ROIDrawer):
    """Freeform circular brush. Mask lives at canvas resolution."""

    def __init__(self, radius=3, fill_alpha=0.45, fill_color=(0, 180, 255),
                 cursor_color=(0, 255, 255)):
        self.radius = max(1, int(radius))
        self.fill_alpha = fill_alpha
        self.fill_color = np.array(fill_color, dtype=np.float32)
        self.cursor_color = cursor_color
        self.canvas_h = 0
        self.canvas_w = 0
        self.mask = np.zeros((1, 1), dtype=np.uint8)
        self._mouse_down = None  # None | 'L' | 'R'
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

    def load_existing(self, mask_np):
        resized = cv2.resize(mask_np.astype(np.float32),
                             (self.canvas_w, self.canvas_h),
                             interpolation=cv2.INTER_AREA)
        self.mask = (resized > 0.5).astype(np.uint8)

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
        # Erase = Shift + L-click (Shift state is read live, so toggling Shift
        # mid-drag switches the stroke between paint and erase).
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


class GtTileROIDrawer(BrushROIDrawer):
    """ROI drawer that shows the ground-truth mask under an N×N tile grid.

    Left-click a tile to copy GT=1 pixels in that tile into the ROI; right-click
    to clear ROI within that tile. Brush behavior from the parent is unused."""

    def __init__(self, grid=8, gt_color=(0, 220, 0), gt_alpha=0.35,
                 grid_color=(255, 255, 0), **brush_kw):
        super().__init__(**brush_kw)
        self.grid = int(grid)
        self.gt_color = np.array(gt_color, dtype=np.float32)
        self.gt_alpha = float(gt_alpha)
        self.grid_color = grid_color
        self.gt_mask = None  # uint8 {0,1} at canvas size

    def reset(self, canvas_h, canvas_w):
        super().reset(canvas_h, canvas_w)
        self.gt_mask = None

    def set_gt(self, gt_np):
        """gt_np: (H, W) array in {0,1}, original resolution. None clears."""
        if gt_np is None:
            self.gt_mask = None
            return
        resized = cv2.resize(gt_np.astype(np.float32),
                             (self.canvas_w, self.canvas_h),
                             interpolation=cv2.INTER_NEAREST)
        self.gt_mask = (resized > 0.5).astype(np.uint8)

    def _tile_extent(self, ix, iy):
        ch, cw = self.canvas_h, self.canvas_w
        g = self.grid
        x0 = int(round(ix * cw / g));  x1 = int(round((ix + 1) * cw / g))
        y0 = int(round(iy * ch / g));  y1 = int(round((iy + 1) * ch / g))
        return y0, y1, x0, x1

    def _apply_tile(self, x, y, add):
        if self.gt_mask is None:
            return
        g = self.grid
        ix = max(0, min(g - 1, int(x * g / self.canvas_w)))
        iy = max(0, min(g - 1, int(y * g / self.canvas_h)))
        y0, y1, x0, x1 = self._tile_extent(ix, iy)
        if add:
            self.mask[y0:y1, x0:x1] = np.maximum(
                self.mask[y0:y1, x0:x1], self.gt_mask[y0:y1, x0:x1]
            )
        else:
            self.mask[y0:y1, x0:x1] = 0

    def on_mouse(self, event, x, y, flags):
        """Side-by-side: GT panel (left) is tile-copy; ROI panel (right) is brush.
        Shift = erase / clear (in place of right-click, which collides with WM menus)."""
        self._cursor = (int(x), int(y))
        if event == cv2.EVENT_LBUTTONUP:
            self._mouse_down = None
            self._last_xy = None
            return
        w = self.canvas_w
        if x < w:
            # GT panel — L-click copies GT in tile; Shift+L-click clears the tile.
            erase = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
            if event == cv2.EVENT_LBUTTONDOWN:
                self._apply_tile(x, y, add=not erase)
            elif event == cv2.EVENT_MOUSEMOVE and self._mouse_down is not None:
                # Cursor crossed into GT mid-stroke; break the brush stroke so the
                # next stroke on the ROI panel starts fresh instead of long-lining.
                self._last_xy = None
        else:
            # ROI panel — delegate to brush behavior, translating x into panel coords
            super().on_mouse(event, x - w, y, flags)
            # super overwrote _cursor with panel-relative coords; restore the global
            # coord so render_overlay can tell the cursor is on the ROI panel and
            # draw the brush circle.
            self._cursor = (int(x), int(y))

    def _label(self, panel, text):
        cv2.rectangle(panel, (0, 0), (60, 20), (0, 0, 0), -1)
        cv2.putText(panel, text, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (220, 220, 220), 1, cv2.LINE_AA)

    def render_overlay(self, canvas_bgr):
        h, w = canvas_bgr.shape[:2]

        # --- LEFT: GT panel (interactive) — global image + GT overlay + grid ---
        gt_panel = canvas_bgr.astype(np.float32)
        if self.gt_mask is not None:
            gt = self.gt_mask.astype(bool)
            gt_layer = np.tile(self.gt_color, (h, w, 1))
            alpha = np.where(gt[..., None], self.gt_alpha, 0.0).astype(np.float32)
            gt_panel = gt_panel * (1 - alpha) + gt_layer * alpha
        gt_panel = np.clip(gt_panel, 0, 255).astype(np.uint8)
        for i in range(1, self.grid):
            xg = int(round(i * w / self.grid))
            cv2.line(gt_panel, (xg, 0), (xg, h - 1), self.grid_color, 1, cv2.LINE_AA)
            yg = int(round(i * h / self.grid))
            cv2.line(gt_panel, (0, yg), (w - 1, yg), self.grid_color, 1, cv2.LINE_AA)
        if self._cursor is not None and 0 <= self._cursor[0] < w:
            g = self.grid
            ix = max(0, min(g - 1, int(self._cursor[0] * g / w)))
            iy = max(0, min(g - 1, int(self._cursor[1] * g / h)))
            y0, y1, x0, x1 = self._tile_extent(ix, iy)
            cv2.rectangle(gt_panel, (x0, y0), (x1 - 1, y1 - 1), self.cursor_color, 1)

        # --- RIGHT: ROI panel (brush-interactive) — global + ROI overlay + brush cursor ---
        roi_panel = canvas_bgr.astype(np.float32)
        roi = self.mask.astype(bool)
        roi_layer = np.tile(self.fill_color, (h, w, 1))
        alpha = np.where(roi[..., None], self.fill_alpha, 0.0).astype(np.float32)
        roi_panel = roi_panel * (1 - alpha) + roi_layer * alpha
        roi_panel = np.clip(roi_panel, 0, 255).astype(np.uint8)
        if self._cursor is not None and self._cursor[0] >= w:
            bx = self._cursor[0] - w
            by = self._cursor[1]
            cv2.circle(roi_panel, (bx, by), self.radius, self.cursor_color, 1,
                       lineType=cv2.LINE_AA)

        self._label(gt_panel,  'GT - click tiles')
        self._label(roi_panel, 'ROI - draw here')
        out = np.hstack([gt_panel, roi_panel])
        cv2.line(out, (w, 0), (w, h - 1), (200, 200, 200), 1)
        return out


def _global_to_bgr(tensor):
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


KEYBINDS = [
    ('L-click / drag', 'paint ROI',
     'Hold left mouse button and drag to paint the ROI in orange.'),
    ('Shift + L-drag', 'erase ROI',
     'Hold Shift and left-click / drag to erase. Use this on a filled mask to carve out islands / holes. In --showgt mode, Shift+L-click on a GT tile clears the ROI inside that tile.'),
    ('- / =',          'shrink / grow brush',
     'Adjust brush radius. Step size scales with current radius.'),
    ('f',              'flood-fill',
     'Fill enclosed regions inside the drawn outline. Shows side-by-side preview; y to accept, n to reject.'),
    ('y / n',          'fill confirm',
     'During flood-fill preview only: y commits the filled mask, n / ESC discards the fill and keeps the outline.'),
    ('s',              'save + next',
     'Save current ROI to the next available slot (roi_<stem>.pt, then roi_<stem>_1.pt, ...) and advance to next slide. With --overwrite, always writes the primary roi_<stem>.pt.'),
    ('a',              'save + stay',
     'Save current ROI to the next available slot and clear the canvas, staying on this slide so you can draw another ROI for the same image.'),
    ('x',              'skip + next',
     'Advance to the next slide without saving. In-progress painting on this slide is discarded.'),
    ('c',              'clear',
     'Clear all painting on the current slide. Does not delete saved files.'),
    ('b',              'back',
     'Reload the previous slide. In-progress unsaved changes on the current slide are lost.'),
    ('[ / ]',          'gamma -/+',
     'Darken / brighten preview via gamma LUT. Display only — does not affect saved ROI.'),
    ('\\',             'reset gamma',
     'Reset preview gamma to 1.0.'),
    ('q / ESC',        'quit',
     'Exit the tool. Current slide is not auto-saved — press s first if you want to keep it.'),
]


def _wrap_text(text, max_chars=42):
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
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    pad = 10
    row_h = 24
    key_col_w = 160
    width = 520
    height = pad * 2 + row_h * len(KEYBINDS)
    img = np.full((height, width, 3), 30, dtype=np.uint8)
    for idx, (key, short, _) in enumerate(KEYBINDS):
        y0 = pad + idx * row_h
        if idx == hover_row:
            cv2.rectangle(img, (0, y0), (width, y0 + row_h), (60, 60, 90), -1)
        baseline = y0 + row_h - 7
        cv2.putText(img, key, (pad, baseline), font, scale, (180, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(img, short, (pad + key_col_w, baseline), font, scale, (220, 220, 220), 1, cv2.LINE_AA)
    if hover_row is not None and cursor_xy is not None:
        long_text = KEYBINDS[hover_row][2]
        lines = _wrap_text(long_text)
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


def _fill_holes(mask):
    """Fill enclosed regions inside a binary outline mask (uint8 {0,1})."""
    pad = cv2.copyMakeBorder(mask.astype(np.uint8), 1, 1, 1, 1,
                             cv2.BORDER_CONSTANT, value=0)
    flood = pad.copy()
    cv2.floodFill(flood, None, (0, 0), 2)
    holes = (flood == 0).astype(np.uint8)
    return (pad | holes)[1:-1, 1:-1]


def _apply_gamma(img_bgr, gamma):
    """Return a gamma-adjusted uint8 copy. gamma > 1 brightens, gamma < 1 darkens."""
    if abs(gamma - 1.0) < 1e-3:
        return img_bgr
    inv = 1.0 / max(gamma, 1e-3)
    lut = np.clip(((np.arange(256) / 255.0) ** inv) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img_bgr, lut)


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
        cv2.putText(img, ln, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 2)


def _next_roi_path(image_dir, stem, overwrite):
    """Return the path the next save should write to.

    overwrite=True  -> always primary roi_<stem>.pt (replace it)
    overwrite=False -> first nonexistent of roi_<stem>.pt, roi_<stem>_1.pt, ..."""
    primary = image_dir / stem / f'roi_{stem}.pt'
    if overwrite or not primary.exists():
        return primary
    i = 1
    while True:
        p = image_dir / stem / f'roi_{stem}_{i}.pt'
        if not p.exists():
            return p
        i += 1


def annotate(data_dir, brush_radius, canvas_size, overwrite, start_from,
             showgt=False, add=False, gt_grid=8):
    image_dir = Path(data_dir) / 'images'
    mask_dir  = Path(data_dir) / 'masks'
    stems = sorted(p.parent.name for p in image_dir.glob('*/global.pt'))
    if not stems:
        raise SystemExit(f'no global.pt under {image_dir}')

    i = 0
    if start_from:
        if start_from not in stems:
            raise SystemExit(f'--start-from {start_from} not found under {image_dir}')
        i = stems.index(start_from)

    if overwrite:
        wiped = 0
        for p in image_dir.glob('*/roi_*.pt'):
            p.unlink()
            wiped += 1
        print(f'[--overwrite] deleted {wiped} existing roi_*.pt files under {image_dir}')

    if showgt:
        drawer = GtTileROIDrawer(grid=gt_grid, radius=brush_radius)
    else:
        drawer = BrushROIDrawer(radius=brush_radius)
    win = 'annotate_globals'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 2 * canvas_size if showgt else canvas_size, canvas_size)
    cv2.setMouseCallback(win, lambda e, x, y, f, _u: drawer.on_mouse(e, x, y, f))

    keys_win = 'keybinds'
    keys_state = {'hover': None, 'cursor': None}

    def _on_keys_mouse(event, x, y, _flags, _u):
        if event == cv2.EVENT_MOUSEMOVE:
            row_h = 24
            pad = 10
            idx = (y - pad) // row_h
            keys_state['hover'] = int(idx) if 0 <= idx < len(KEYBINDS) else None
            keys_state['cursor'] = (int(x), int(y))
        elif event == cv2.EVENT_MOUSELEAVE:
            keys_state['hover'] = None
            keys_state['cursor'] = None

    cv2.namedWindow(keys_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(keys_win, 520, 60 + 24 * len(KEYBINDS))
    cv2.setMouseCallback(keys_win, _on_keys_mouse)

    BRUSH_MAX = 50
    cv2.createTrackbar('brush', keys_win, max(1, min(BRUSH_MAX, drawer.radius)),
                       BRUSH_MAX, lambda v: drawer.set_radius(max(1, v)))

    n_saved = n_skipped = 0
    current_stem = None
    canvas = None
    H = W = 0
    gamma = 1.0
    pending_fill = None  # uint8 mask awaiting y/n confirmation

    def _load_slide(stem):
        nonlocal canvas, H, W
        img = torch.load(image_dir / stem / 'global.pt', weights_only=True)
        _, H, W = img.shape
        bgr = _global_to_bgr(img)
        canvas = cv2.resize(bgr, (canvas_size, canvas_size),
                            interpolation=cv2.INTER_LINEAR)
        drawer.reset(canvas_size, canvas_size)
        if showgt:
            gt_path = mask_dir / stem / 'global.pt'
            if gt_path.exists():
                gt = torch.load(gt_path, weights_only=True).squeeze(0).numpy()
                drawer.set_gt(gt)
            else:
                print(f'  [warn] --showgt: no GT mask at {gt_path}')
                drawer.set_gt(None)

    while 0 <= i < len(stems):
        stem = stems[i]
        primary_path = image_dir / stem / f'roi_{stem}.pt'

        if primary_path.exists() and not add:
            n_skipped += 1
            i += 1
            continue

        if stem != current_stem:
            _load_slide(stem)
            current_stem = stem

        gamma_canvas = _apply_gamma(canvas, gamma)
        if pending_fill is not None:
            left = gamma_canvas.copy()
            saved_mask = drawer.mask
            drawer.mask = pending_fill
            right = drawer.render_overlay(gamma_canvas)
            drawer.mask = saved_mask
            view = np.hstack([left, right])
            _draw_banner(view, [
                f'[{i+1}/{len(stems)}] {stem}   FILL PREVIEW',
                'y = accept fill   n / ESC = reject',
            ])
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
        _draw_banner(view, f'[{i+1}/{len(stems)}] {stem}   gamma={gamma:.2f}   brush={drawer.radius}px')
        cv2.imshow(win, view)
        cv2.imshow(keys_win, _render_keybinds(keys_state['hover'], keys_state['cursor']))
        key = cv2.waitKey(15) & 0xFF

        if key in (ord('q'), 27):
            break
        if key == ord('c'):
            drawer.clear()
        elif key == ord('f'):
            pending_fill = _fill_holes(drawer.mask)
        elif key == ord('b'):
            i = max(0, i - 1)
            current_stem = None
        elif key == ord('x'):
            confirm_view = drawer.render_overlay(gamma_canvas).copy()
            _draw_banner(confirm_view, [
                f'[{i+1}/{len(stems)}] {stem}   SKIP THIS SLIDE?',
                'y = skip without saving   n / ESC = cancel',
            ])
            cv2.imshow(win, confirm_view)
            while True:
                ck = cv2.waitKey(15) & 0xFF
                if ck == ord('y'):
                    n_skipped += 1
                    i += 1
                    current_stem = None
                    break
                if ck in (ord('n'), 27):
                    break
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
        elif key == ord('s'):
            out_path = _next_roi_path(image_dir, stem, overwrite)
            mask_tensor = drawer.get_mask(H, W)
            torch.save(mask_tensor, out_path)
            print(f'  saved {out_path.name}')
            n_saved += 1
            i += 1
            current_stem = None
        elif key == ord('a'):
            out_path = _next_roi_path(image_dir, stem, overwrite=False)
            mask_tensor = drawer.get_mask(H, W)
            torch.save(mask_tensor, out_path)
            print(f'  saved {out_path.name} (stay)')
            n_saved += 1
            # stay on this stem with a fresh canvas; preserve GT overlay if present
            drawer.reset(canvas_size, canvas_size)
            if showgt:
                gt_path = mask_dir / stem / 'global.pt'
                if gt_path.exists():
                    gt = torch.load(gt_path, weights_only=True).squeeze(0).numpy()
                    drawer.set_gt(gt)

    cv2.destroyAllWindows()
    print(f'done: saved={n_saved} skipped={n_skipped} total={len(stems)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', required=True,
                        help='dataset root (contains images/<stem>/global.pt)')
    parser.add_argument('--brush-radius', type=int, default=3,
                        help='initial brush radius in canvas pixels (default 3)')
    parser.add_argument('--canvas-size', type=int, default=768,
                        help='display window size in pixels (default 768)')
    parser.add_argument('--overwrite', action='store_true',
                        help='DELETE every existing roi_*.pt under the data dir at startup, then annotate from scratch')
    parser.add_argument('--start-from', default=None,
                        help='stem to resume from')
    parser.add_argument('--add', action='store_true',
                        help='do not skip slides that already have a primary ROI; save new ROIs to roi_<stem>_N.pt')
    parser.add_argument('--showgt', action='store_true',
                        help='show ground-truth mask under a tile grid; click tiles to copy GT foreground into the ROI')
    parser.add_argument('--gt-grid', type=int, default=8,
                        help='grid size (cells per side) used by --showgt (default 8)')
    annotate(**vars(parser.parse_args()))
