"""
Rapid-fire ROI annotation for every global.pt in a preprocessed data dir.

Each slide gets a single ROI written to:
    <data_dir>/images/<stem>/roi_<stem>.pt      shape (1, H, W) float32 in {0,1}

UI is swappable: `GridROIDrawer` implements the `ROIDrawer` interface; swap in
a `BrushROIDrawer` (or similar) later without touching the main loop.

Usage:
    python annotate_globals.py --data-dir data/[collapsed]64_tiles_per_image
    python annotate_globals.py --data-dir ... --grid 32 --canvas-size 768
    python annotate_globals.py --data-dir ... --overwrite
    python annotate_globals.py --data-dir ... --start-from <stem>

Keys:
    L-click / drag     paint ROI cells
    R-click / drag     erase ROI cells
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


class GridROIDrawer(ROIDrawer):
    def __init__(self, grid=32, fill_alpha=0.45, fill_color=(0, 180, 255),
                 line_color=(80, 80, 80)):
        self.grid = grid
        self.fill_alpha = fill_alpha
        self.fill_color = np.array(fill_color, dtype=np.float32)
        self.line_color = line_color
        self.mask = np.zeros((grid, grid), dtype=bool)
        self.canvas_h = 0
        self.canvas_w = 0
        self._mouse_down = None  # None | 'L' | 'R'

    def reset(self, canvas_h, canvas_w):
        self.canvas_h = canvas_h
        self.canvas_w = canvas_w
        self.mask = np.zeros((self.grid, self.grid), dtype=bool)
        self._mouse_down = None

    def clear(self):
        self.mask[:] = False

    def load_existing(self, mask_np):
        small = cv2.resize(mask_np.astype(np.float32), (self.grid, self.grid),
                           interpolation=cv2.INTER_AREA)
        self.mask = small > 0.5

    def _cell(self, x, y):
        if self.canvas_w == 0 or self.canvas_h == 0:
            return 0, 0
        gx = int(x / self.canvas_w * self.grid)
        gy = int(y / self.canvas_h * self.grid)
        gx = max(0, min(self.grid - 1, gx))
        gy = max(0, min(self.grid - 1, gy))
        return gy, gx

    def on_mouse(self, event, x, y, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_down = 'L'
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._mouse_down = 'R'
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self._mouse_down = None
            return
        if self._mouse_down is None:
            return
        gy, gx = self._cell(x, y)
        self.mask[gy, gx] = (self._mouse_down == 'L')

    def render_overlay(self, canvas_bgr):
        h, w = canvas_bgr.shape[:2]
        filled = cv2.resize(self.mask.astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        out = canvas_bgr.astype(np.float32)
        fill_layer = np.tile(self.fill_color, (h, w, 1))
        alpha = np.where(filled[..., None], self.fill_alpha, 0.0).astype(np.float32)
        out = out * (1 - alpha) + fill_layer * alpha
        out = np.clip(out, 0, 255).astype(np.uint8)
        for i in range(1, self.grid):
            y = int(i * h / self.grid)
            x = int(i * w / self.grid)
            cv2.line(out, (0, y), (w, y), self.line_color, 1)
            cv2.line(out, (x, 0), (x, h), self.line_color, 1)
        return out

    def get_mask(self, target_h, target_w):
        full = cv2.resize(self.mask.astype(np.uint8), (target_w, target_h),
                          interpolation=cv2.INTER_NEAREST).astype(np.float32)
        return torch.from_numpy(full).unsqueeze(0)


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


def annotate(data_dir, grid, canvas_size, overwrite, start_from):
    image_dir = Path(data_dir) / 'images'
    stems = sorted(p.parent.name for p in image_dir.glob('*/global.pt'))
    if not stems:
        raise SystemExit(f'no global.pt under {image_dir}')

    i = 0
    if start_from:
        if start_from not in stems:
            raise SystemExit(f'--start-from {start_from} not found under {image_dir}')
        i = stems.index(start_from)

    drawer = GridROIDrawer(grid=grid)
    win = 'annotate_globals'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, canvas_size, canvas_size)
    cv2.setMouseCallback(win, lambda e, x, y, f, _u: drawer.on_mouse(e, x, y, f))

    n_saved = n_skipped = 0
    current_stem = None
    canvas = None
    H = W = 0
    gamma = 1.0

    def _load_slide(stem):
        nonlocal canvas, H, W
        img = torch.load(image_dir / stem / 'global.pt', weights_only=True)
        _, H, W = img.shape
        bgr = _global_to_bgr(img)
        canvas = cv2.resize(bgr, (canvas_size, canvas_size),
                            interpolation=cv2.INTER_LINEAR)
        drawer.reset(canvas_size, canvas_size)
        roi_path = image_dir / stem / f'roi_{stem}.pt'
        if roi_path.exists():
            existing = torch.load(roi_path, weights_only=True).squeeze(0).numpy()
            drawer.load_existing(existing)

    while 0 <= i < len(stems):
        stem = stems[i]
        roi_path = image_dir / stem / f'roi_{stem}.pt'

        if roi_path.exists() and not overwrite:
            n_skipped += 1
            i += 1
            continue

        if stem != current_stem:
            _load_slide(stem)
            current_stem = stem

        view = drawer.render_overlay(_apply_gamma(canvas, gamma))
        _draw_banner(view, [
            f'[{i+1}/{len(stems)}] {stem}   gamma={gamma:.2f}',
            'L=paint  R=erase  s=save+next  c=clear  b=back  [=darker  ]=brighter  \\=reset gamma  q=quit',
        ])
        cv2.imshow(win, view)
        key = cv2.waitKey(15) & 0xFF

        if key in (ord('q'), 27):
            break
        if key == ord('c'):
            drawer.clear()
        elif key == ord('b'):
            i = max(0, i - 1)
            current_stem = None
        elif key == ord('['):
            gamma = max(0.1, round(gamma - 0.1, 2))
        elif key == ord(']'):
            gamma = min(5.0, round(gamma + 0.1, 2))
        elif key == ord('\\'):
            gamma = 1.0
        elif key == ord('s'):
            mask_tensor = drawer.get_mask(H, W)
            torch.save(mask_tensor, roi_path)
            n_saved += 1
            i += 1
            current_stem = None

    cv2.destroyAllWindows()
    print(f'done: saved={n_saved} skipped={n_skipped} total={len(stems)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', required=True,
                        help='dataset root (contains images/<stem>/global.pt)')
    parser.add_argument('--grid', type=int, default=32,
                        help='grid cells per side (default 32)')
    parser.add_argument('--canvas-size', type=int, default=768,
                        help='display window size in pixels (default 768)')
    parser.add_argument('--overwrite', action='store_true',
                        help='re-annotate slides that already have an ROI')
    parser.add_argument('--start-from', default=None,
                        help='stem to resume from')
    annotate(**vars(parser.parse_args()))
