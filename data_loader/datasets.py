from pathlib import Path
import json
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


def _geo_augment(*tensors, pos, flags):
    """Back-compat: sample one set of geo params and apply to all tensors + pos."""
    params = _sample_geo_params(flags)
    tensors = tuple(_apply_geo_tensor(t, params) for t in tensors)
    pos = _apply_geo_pos(pos, params)
    return (*tensors, pos)


class GlobalDataset(Dataset):
    """
    Loads only the downsampled global.pt image + corresponding global mask.

    Each patient directory must contain:
        data/images/<stem>/global.pt  — (3, H, W) float32
        data/masks/<stem>/global.pt   — (1, H, W) float32

    Returns (image, mask).
    """
    def __init__(self, data_dir):
        data_dir = Path(data_dir)
        image_dir = data_dir / 'images'
        mask_dir  = data_dir / 'masks'

        self.pairs = []
        for img_path in sorted(image_dir.glob('*/global.pt')):
            mask_path = mask_dir / img_path.parent.name / 'global.pt'
            if mask_path.exists():
                self.pairs.append((img_path, mask_path))

        assert len(self.pairs) > 0, f"No global.pt pairs found in {data_dir}"

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = torch.load(img_path,  weights_only=True)
        mask  = torch.load(mask_path, weights_only=True)
        return image, mask


class LocalDataset(Dataset):
    """
    Loads local patch tiles + corresponding masks only — no global context.
    Returns (image, mask).
    """
    def __init__(self, data_dir):
        data_dir = Path(data_dir)
        self.image_dir = data_dir / 'images'
        self.mask_dir  = data_dir / 'masks'

        self.items = []
        for coords_file in sorted(self.image_dir.glob('*/coordinates.json')):
            stem = coords_file.parent.name
            with open(coords_file) as f:
                entries = json.load(f)
            for entry in entries:
                patch_path = coords_file.parent / entry['filename']
                self.items.append((patch_path, stem))

        assert len(self.items) > 0, f"No patches found via coordinates.json in {self.image_dir}"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        patch_path, stem = self.items[idx]
        image = torch.load(patch_path, weights_only=True)

        mask_path = self.mask_dir / stem / patch_path.name
        if mask_path.exists():
            mask = torch.load(mask_path, weights_only=True)
        else:
            mask = torch.zeros(1, image.shape[-2], image.shape[-1])

        return image, mask


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
                 roi_exclusion=False, roi_exclusion_radius=0.05):
        data_dir = Path(data_dir)
        self.image_dir = data_dir / 'images'
        self.mask_dir  = data_dir / 'masks'
        self.aug_flags = {'hflip': hflip, 'vflip': vflip, 'rot90': rot90}
        self.augment = any(v > 0 for v in self.aug_flags.values())
        self.require_roi = require_roi
        self.roi_exclusion = roi_exclusion
        self.roi_exclusion_radius = roi_exclusion_radius

        # Photometric — applied to local only (add blur, erasure, etc. here)
        self.photo_transform = None

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
                primary = coords_file.parent / f'roi_{stem}.pt'
                if not primary.exists():
                    raise FileNotFoundError(
                        f"require_roi=True but missing ROI for stem '{stem}': {primary}. "
                        f"Run annotate_globals.py --data-dir {data_dir}."
                    )
                self.stem_to_roi[stem] = sorted(
                    coords_file.parent.glob(f'roi_{stem}*.pt')
                )

        assert len(self.items) > 0, f"No patches found via coordinates.json in {self.image_dir}"

        if self.roi_exclusion:
            assert self.require_roi, "roi_exclusion requires require_roi=True"
            self._filter_by_roi_exclusion()

        assert len(self.items) > 0, "No tiles remain after roi_exclusion filtering"

    def _filter_by_roi_exclusion(self):
        import cv2
        stem_dist = {}
        for stem, roi_paths in self.stem_to_roi.items():
            # Union of all ROIs: keep a tile that is near any of the available ROIs.
            roi_union = None
            H_roi = W_roi = 0
            for roi_path in roi_paths:
                roi = torch.load(roi_path, weights_only=True).squeeze(0).numpy()
                roi_bin = (roi > 0.5).astype(np.uint8)
                H_roi, W_roi = roi_bin.shape
                roi_union = roi_bin if roi_union is None else np.maximum(roi_union, roi_bin)
            if roi_union is None or roi_union.max() == 0:
                stem_dist[stem] = None
            else:
                dist = cv2.distanceTransform(1 - roi_union, cv2.DIST_L2, 5)
                stem_dist[stem] = (dist, H_roi, W_roi)

        kept = []
        for item in self.items:
            _, stem, entry = item
            if stem not in stem_dist or stem_dist[stem] is None:
                kept.append(item)
                continue
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
        # pos must describe the full context window the local patch spans, not the
        # inner tile — the model bottleneck encodes the whole context patch, so the
        # global/ROI crop has to register against that same extent. Reduces to the
        # inner tile when context_size == tile_size.
        ctx = entry.get('context_size', w)
        pad = (ctx - w) // 2
        cy, cx = y - pad, x - pad
        pos = torch.tensor([cy / H, cx / W, (cy + ctx) / H, (cx + ctx) / W], dtype=torch.float32)

        mask_path = self.mask_dir / stem / patch_path.name
        if mask_path.exists():
            mask = torch.load(mask_path, weights_only=True)
        else:
            mask = torch.zeros(1, local.shape[-2], local.shape[-1])

        roi = None
        if self.require_roi:
            roi = torch.load(random.choice(self.stem_to_roi[stem]), weights_only=True)

        if self.augment:
            if roi is not None:
                local, roi, mask, pos = _geo_augment(
                    local, roi, mask, pos=pos, flags=self.aug_flags,
                )
            else:
                local, mask, pos = _geo_augment(
                    local, mask, pos=pos, flags=self.aug_flags,
                )
        if self.photo_transform is not None:
            local = self.photo_transform(local)

        if roi is not None:
            return local, pos, roi, mask
        return local, pos, mask

    def foreground_indices(self, min_ratio=0.05):
        """Partition tile indices into (fg, bg) by mask foreground fraction."""
        fg, bg = [], []
        print(f"[LocalGlobalDataset] Scanning {len(self.items)} tiles "
              f"for foreground ratio >= {min_ratio}...")
        for i, (patch_path, stem, _) in enumerate(self.items):
            mask_path = self.mask_dir / stem / patch_path.name
            if not mask_path.exists():
                bg.append(i)
                continue
            mask = torch.load(mask_path, weights_only=True)
            frac = (mask > 0.5).float().mean().item()
            (fg if frac >= min_ratio else bg).append(i)
        total = len(self.items)
        print(f"[LocalGlobalDataset] Foreground tiles: "
              f"{len(fg)}/{total} ({100*len(fg)/total:.1f}%)")
        return fg, bg


