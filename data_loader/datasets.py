from pathlib import Path
import json
import math
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


def _sample_geo_params(flags):
    """Sample a per-view geometric aug as (hflip, vflip, k)."""
    hflip = 1 if random.random() < flags.get('hflip', 0.0) else 0
    vflip = 1 if random.random() < flags.get('vflip', 0.0) else 0
    k = random.randint(1, 3) if random.random() < flags.get('rot90', 0.0) else 0
    return (hflip, vflip, k)


def _apply_geo_tensor(t, params):
    """Apply geo params (hflip, vflip, k) to a (C, H, W) tensor.
    Op order: hflip -> vflip -> rot90_k (must match _apply_geo_pos)."""
    hflip, vflip, k = params
    if hflip:
        t = torch.flip(t, [-1])
    if vflip:
        t = torch.flip(t, [-2])
    if k:
        t = torch.rot90(t, k, [-2, -1])
    return t


def _apply_geo_pos(pos, params):
    """Apply geo params to normalised [y0, x0, y1, x1] pos."""
    hflip, vflip, k = params
    y0, x0, y1, x1 = pos[0], pos[1], pos[2], pos[3]
    if hflip:
        x0, x1 = 1 - x1, 1 - x0
    if vflip:
        y0, y1 = 1 - y1, 1 - y0
    for _ in range(k):
        y0, x0, y1, x1 = 1 - x1, y0, 1 - x0, y1
    return torch.tensor([y0, x0, y1, x1], dtype=pos.dtype)


def _brightness_jitter(lo, hi):
    """Exposure jitter, log-uniform in [lo, hi].

    Multiplicative and log-uniform because exposure scales the signal and spans
    ~10x here (p99.9 runs 0.072 to 0.702 across the twenty sections). The range
    is asymmetric because the directions are not equivalent: tiles live in
    [0, 1], so darkening is lossless while brightening saturates against the
    ceiling -- the brightest section clips at ~1.4x, the dimmest not until ~14x.
    """
    log_lo, log_hi = math.log(lo), math.log(hi)

    def apply(t):
        return (t * math.exp(random.uniform(log_lo, log_hi))).clamp(max=1.0)

    return apply


def _geo_augment(*tensors, pos, flags):
    """Back-compat: sample one set of geo params and apply to all tensors + pos."""
    params = _sample_geo_params(flags)
    tensors = tuple(_apply_geo_tensor(t, params) for t in tensors)
    pos = _apply_geo_pos(pos, params)
    return (*tensors, pos)


class LocalGlobalDataset(Dataset):
    """
    Loads a local patch tile + its mask, with the tile's position on the slide.

    Index is built by reading coordinates.json files written by preprocess.py.
    All files for a given image live together in data_dir/<subdir>/<stem>/:
        patch_YYYY_XXXX.pt   — (C, H, W) float32  local tile
        coordinates.json     — patch index with stem, x, y, image_W, image_H

    Returns (local, pos, mask) where pos = [y0, x0, y1, x1] normalised.
    If require_roi=True, also loads a per-stem ROI mask written by
    annotate_globals.py (roi_<stem>.pt, shape (1, H, W)) and returns it as
    (local, pos, roi, mask).

    The whole-slide global.pt is NOT read here. It was passed to the model as
    `glob` for a global-context branch that no longer exists, and its mask
    counterpart was the auxiliary head's target; both were loaded per tile and
    discarded. global.pt is still written by preprocess.py and still read
    directly from disk by annotate_globals.py and the trainer's split previews.
    """
    def __init__(self, data_dir, hflip=0.0, vflip=0.0, rot90=0.0,
                 require_roi=False,
                 roi_exclusion=False, roi_exclusion_radius=0.05,
                 brightness=None):
        data_dir = Path(data_dir)
        self.image_dir = data_dir / 'images'
        self.mask_dir  = data_dir / 'masks'
        self.aug_flags = {'hflip': hflip, 'vflip': vflip, 'rot90': rot90}
        self.augment = any(v > 0 for v in self.aug_flags.values())
        self.require_roi = require_roi
        self.roi_exclusion = roi_exclusion
        self.roi_exclusion_radius = roi_exclusion_radius

        # Photometric — applied to local only (add blur, erasure, etc. here).
        # brightness is a [lo, hi] multiplicative range; None or absent disables.
        self.photo_transform = _brightness_jitter(*brightness) if brightness else None

        self.items = []
        self.stem_to_roi = {}
        for coords_file in sorted(self.image_dir.glob('*/coordinates.json')):
            stem = coords_file.parent.name
            with open(coords_file) as f:
                entries = json.load(f)
            for entry in entries:
                patch_path = coords_file.parent / entry['filename']
                self.items.append((patch_path, stem, entry))
            if self.require_roi:
                # Exactly one ROI per slide: roi_<stem>.pt, no suffix. Any
                # roi_<stem>_N.pt variants present on disk are ignored.
                roi_path = coords_file.parent / f'roi_{stem}.pt'
                if not roi_path.exists():
                    raise FileNotFoundError(
                        f"require_roi=True but missing ROI for stem '{stem}': {roi_path}. "
                        f"Run annotate_globals.py --data-dir {data_dir}."
                    )
                self.stem_to_roi[stem] = roi_path

        assert len(self.items) > 0, f"No patches found via coordinates.json in {self.image_dir}"

        if self.roi_exclusion:
            assert self.require_roi, "roi_exclusion requires require_roi=True"
            self._filter_by_roi_exclusion()

        assert len(self.items) > 0, "No tiles remain after roi_exclusion filtering"

    def _filter_by_roi_exclusion(self):
        import cv2
        stem_dist = {}
        for stem, roi_path in self.stem_to_roi.items():
            roi_bin = (torch.load(roi_path, weights_only=True).squeeze(0).numpy() > 0.5)
            roi_bin = roi_bin.astype(np.uint8)
            if roi_bin.max() == 0:
                raise ValueError(
                    f"ROI for stem '{stem}' is empty ({roi_path}). Filtering against it "
                    f"would keep every tile for this slide instead of the ROI-adjacent "
                    f"ones, silently skewing the split toward background.")
            H_roi, W_roi = roi_bin.shape
            stem_dist[stem] = (cv2.distanceTransform(1 - roi_bin, cv2.DIST_L2, 5),
                               H_roi, W_roi)

        kept = []
        for item in self.items:
            _, stem, entry = item
            dist, H_roi, W_roi = stem_dist[stem]
            image_H, image_W = entry['image_H'], entry['image_W']
            y, x = entry['y'], entry['x']
            h, w = entry['height'], entry['width']
            y0_roi = int(np.clip(y / image_H * H_roi, 0, H_roi - 1))
            x0_roi = int(np.clip(x / image_W * W_roi, 0, W_roi - 1))
            y1_roi = int(np.clip((y + h) / image_H * H_roi, 0, H_roi - 1)) + 1
            x1_roi = int(np.clip((x + w) / image_W * W_roi, 0, W_roi - 1)) + 1
            min_dist = dist[y0_roi:y1_roi, x0_roi:x1_roi].min()
            threshold_roi_px = self.roi_exclusion_radius * min(image_H, image_W) * (H_roi / image_H)
            if min_dist <= threshold_roi_px:
                kept.append(item)

        n_before = len(self.items)
        self.items = kept
        print(f"[LocalGlobalDataset] ROI exclusion: kept {len(self.items)}/{n_before} tiles "
              f"(radius={self.roi_exclusion_radius*100:.1f}% of image min-dim)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        patch_path, stem, entry = self.items[idx]

        local = torch.load(patch_path, weights_only=True)

        H, W = entry['image_H'], entry['image_W']
        y, x = entry['y'], entry['x']
        h, w = entry['height'], entry['width']
        # pos is the tile's extent on the slide, normalised, and is what the model
        # uses to crop the ROI to this tile. Image and mask tiles are both saved
        # at tile_size, so the tile IS the whole extent the encoder sees.
        pos = torch.tensor([y / H, x / W, (y + h) / H, (x + w) / W], dtype=torch.float32)

        mask_path = self.mask_dir / stem / patch_path.name
        if mask_path.exists():
            mask = torch.load(mask_path, weights_only=True)
        else:
            mask = torch.zeros(1, local.shape[-2], local.shape[-1])

        roi = None
        if self.require_roi:
            roi = torch.load(self.stem_to_roi[stem], weights_only=True)

        if self.augment:
            if roi is not None:
                local, roi, mask, pos = _geo_augment(
                    local, roi, mask, pos=pos, flags=self.aug_flags,
                )
            else:
                local, mask, pos = _geo_augment(
                    local, mask, pos=pos, flags=self.aug_flags,
                )
            # Inside the augment guard: every evaluation path zeroes the geo
            # flags, so a config carrying `brightness` cannot randomise a
            # held-out score.
            if self.photo_transform is not None:
                local = self.photo_transform(local)

        if roi is not None:
            return local, pos, roi, mask
        return local, pos, mask



