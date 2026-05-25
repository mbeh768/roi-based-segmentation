from pathlib import Path
import json
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset


def _roi_grid_distort(roi, num_steps=5, strength=0.3):
    """Random grid distortion on a (1, H, W) ROI tensor. Decouples ROI from anatomy
    so the model can't memorize a pixel-perfect template."""
    _, H, W = roi.shape
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                            indexing='ij')
    base = torch.stack([xx, yy], dim=-1)  # (H, W, 2)
    cp = torch.randn(2, num_steps, num_steps) * strength
    cp_up = F.interpolate(cp.unsqueeze(0), size=(H, W),
                          mode='bilinear', align_corners=True).squeeze(0)
    base[..., 0] = base[..., 0] + cp_up[0]
    base[..., 1] = base[..., 1] + cp_up[1]
    out = F.grid_sample(roi.unsqueeze(0), base.unsqueeze(0), mode='nearest',
                        padding_mode='border', align_corners=True).squeeze(0)
    return out


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
    Loads local patch + corresponding global image + mask.

    Index is built by reading coordinates.json files written by preprocess.py.
    All files for a given image live together in data_dir/<subdir>/<stem>/:
        global.pt            — (C, H, W) float32  downsampled whole-slide
        patch_YYYY_XXXX.pt   — (C, H, W) float32  local tile
        coordinates.json     — patch index with stem, x, y, image_W, image_H

    Returns (local, glob, pos, mask) where pos = [y0, x0, y1, x1] normalised.
    If require_roi=True, also loads a per-stem ROI mask written by
    annotate_globals.py (roi_<stem>.pt, shape (1, H, W)) and returns it alongside
    the other tensors.
    """
    def __init__(self, data_dir, hflip=0.0, vflip=0.0, rot90=0.0,
                 require_roi=False, roi_jitter=0.0,
                 roi_dropout=0.0, roi_distort=0.0, roi_distort_strength=0.3):
        data_dir = Path(data_dir)
        self.image_dir = data_dir / 'images'
        self.mask_dir  = data_dir / 'masks'
        self.aug_flags = {'hflip': hflip, 'vflip': vflip, 'rot90': rot90}
        self.augment = any(v > 0 for v in self.aug_flags.values())
        self.require_roi = require_roi
        self.roi_jitter = roi_jitter
        self.roi_dropout = roi_dropout
        self.roi_distort = roi_distort
        self.roi_distort_strength = roi_distort_strength

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
                roi_path = coords_file.parent / f'roi_{stem}.pt'
                if not roi_path.exists():
                    raise FileNotFoundError(
                        f"require_roi=True but missing ROI for stem '{stem}': {roi_path}. "
                        f"Run annotate_globals.py --data-dir {data_dir}."
                    )
                self.stem_to_roi[stem] = roi_path

        assert len(self.items) > 0, f"No patches found via coordinates.json in {self.image_dir}"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        patch_path, stem, entry = self.items[idx]

        local = torch.load(patch_path, weights_only=True)
        glob  = torch.load(self.image_dir / stem / 'global.pt', weights_only=True)

        H, W = entry['image_H'], entry['image_W']
        y, x = entry['y'], entry['x']
        h, w = entry['height'], entry['width']
        pos = torch.tensor([y / H, x / W, (y + h) / H, (x + w) / W], dtype=torch.float32)

        mask_path = self.mask_dir / stem / patch_path.name
        if mask_path.exists():
            mask = torch.load(mask_path, weights_only=True)
        else:
            mask = torch.zeros(1, local.shape[-2], local.shape[-1])

        global_mask_path = self.mask_dir / stem / 'global.pt'
        if not global_mask_path.exists():
            raise FileNotFoundError(f"Global mask not found: {global_mask_path}")
        global_mask = torch.load(global_mask_path, weights_only=True)

        roi = None
        if self.require_roi:
            roi = torch.load(self.stem_to_roi[stem], weights_only=True)
            if self.roi_jitter > 0 and random.random() < self.roi_jitter:
                roi = torch.ones_like(roi)
            if self.roi_dropout > 0 and random.random() < self.roi_dropout:
                roi = torch.zeros_like(roi)
            if self.roi_distort > 0 and random.random() < self.roi_distort:
                roi = _roi_grid_distort(roi, strength=self.roi_distort_strength)

        if self.augment:
            if roi is not None:
                local, glob, roi, mask, global_mask, pos = _geo_augment(
                    local, glob, roi, mask, global_mask, pos=pos, flags=self.aug_flags,
                )
            else:
                local, glob, mask, global_mask, pos = _geo_augment(
                    local, glob, mask, global_mask, pos=pos, flags=self.aug_flags,
                )
        if self.photo_transform is not None:
            local = self.photo_transform(local)

        if roi is not None:
            return local, glob, pos, roi, mask, global_mask
        return local, glob, pos, mask, global_mask

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

    def save_visualizations(self, out_dir, n_samples=20):
        """Save before/after augmentation side-by-side previews with pos marked."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prev_augment = self.augment
        self.augment = False

        def _tensor_to_bgr(t):
            t = t.detach().cpu().float()
            lo, hi = t.min(), t.max()
            t = (t - lo) / (hi - lo + 1e-8)
            arr = (t.numpy() * 255).astype(np.uint8)
            if arr.ndim == 3:
                if arr.shape[0] == 1:
                    arr = np.repeat(arr, 3, axis=0)
                arr = arr.transpose(1, 2, 0)  # CHW -> HWC
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return arr

        def _draw_pos(img, pos):
            H, W = img.shape[:2]
            y0, x0, y1, x1 = pos.tolist()
            p1 = (int(x0 * W), int(y0 * H))
            p2 = (int(x1 * W), int(y1 * H))
            return cv2.rectangle(img.copy(), p1, p2, (0, 255, 0), 2)

        def _unpack(sample):
            """Return (local, glob, pos, roi, mask, gmask) regardless of whether
            ROIs are enabled (roi is None when not)."""
            if len(sample) == 6:
                local, glob, pos, roi, mask, gmask = sample
                return local, glob, pos, roi, mask, gmask
            local, glob, pos, mask, gmask = sample
            return local, glob, pos, None, mask, gmask

        n = min(n_samples, len(self.items))
        idxs = random.sample(range(len(self.items)), n)
        for i in idxs:
            # before
            before = _unpack(self[i])
            # after
            self.augment = True
            after = _unpack(self[i])
            self.augment = False

            border_px = 4
            border_color = (0, 255, 0)
            rows = []
            for (local, glob, pos, roi, mask, gmask), label in [
                (before, 'before'),
                (after,  'after'),
            ]:
                g = _draw_pos(_tensor_to_bgr(glob),  pos)
                gm = _draw_pos(_tensor_to_bgr(gmask), pos)
                l = _tensor_to_bgr(local)
                m = _tensor_to_bgr(mask)
                # resize local/mask to match global height
                gh = g.shape[0]
                l = cv2.resize(l, (gh, gh))
                m = cv2.resize(m, (gh, gh))
                panel_tensors = [g, gm, l, m]
                if roi is not None:
                    r = _draw_pos(_tensor_to_bgr(roi), pos)
                    panel_tensors.insert(2, r)  # place ROI next to global mask
                panels = [
                    cv2.copyMakeBorder(p, border_px, border_px, border_px, border_px,
                                       cv2.BORDER_CONSTANT, value=border_color)
                    for p in panel_tensors
                ]
                row = np.concatenate(panels, axis=1)
                cv2.putText(row, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                rows.append(row)
            out = np.concatenate(rows, axis=0)
            cv2.imwrite(str(out_dir / f'sample_{i:04d}.png'), out)

        self.augment = prev_augment
        print(f"[LocalGlobalDataset] Saved {n} visualization(s) to {out_dir}")


class SSLGlobalLocalDataset(Dataset):
    """
    Self-supervised dataset for pretraining _GlobalUNet.

    Returns (aug_global, pos, clean_global, local_target_stack, erase_mask) where:
        aug_global:         (C, H, W) augmented global image (model input)
        pos:                (N, 4) normalised tile regions [y0, x0, y1, x1] for N crops
        clean_global:       (C, H, W) original global image (global reconstruction target)
        local_target_stack: (N, C, crop_size, crop_size) downsampled local tiles
        erase_mask:         (1, H, W) binary mask of erased region

    No masks needed -- works with --nomask preprocessed data.
    """
    def __init__(self, data_dir, crop_size=32,
                 noise_std=0.1, color_scale_range=0.2,
                 blur_kernel=5,
                 p_noise=0.5, p_color=0.5, p_blur=0.3,
                 mask_ratio=0.0, mask_patch_size=16,
                 p_intensity_invert=0.0,
                 hflip=0.0, vflip=0.0, rot90=0.0,
                 num_local_crops=1):
        data_dir = Path(data_dir)
        self.image_dir = data_dir / 'images'
        self.crop_size = crop_size
        self.num_local_crops = num_local_crops
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size
        self.aug_flags = {'hflip': hflip, 'vflip': vflip, 'rot90': rot90}
        self.augment = any(v > 0 for v in self.aug_flags.values())

        self.transform = T.Compose([
            T.RandomApply([T.Lambda(lambda x: x + torch.randn_like(x) * noise_std)], p=p_noise),
            T.RandomApply([T.Lambda(lambda x: x * (1.0 + (torch.rand(x.shape[0], 1, 1) * 2 - 1) * color_scale_range))], p=p_color),
            T.RandomApply([T.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0))], p=p_blur),
            T.RandomApply([T.Lambda(lambda x: -x)], p=p_intensity_invert),
        ])

        self.items = []
        self.stem_to_items = {}
        for coords_file in sorted(self.image_dir.glob('*/coordinates.json')):
            stem = coords_file.parent.name
            with open(coords_file) as f:
                entries = json.load(f)
            stem_items = []
            for entry in entries:
                patch_path = coords_file.parent / entry['filename']
                item = (patch_path, stem, entry)
                self.items.append(item)
                stem_items.append(item)
            self.stem_to_items[stem] = stem_items

        assert len(self.items) > 0, f"No patches found via coordinates.json in {self.image_dir}"

    def __len__(self):
        return len(self.items)

    def _augment(self, img):
        """Apply photometric augs + random rectangular erase to a (C, H, W) tensor.
        Returns (augmented, erase_mask) where erase_mask is (1, H, W) float in {0, 1}."""
        img = self.transform(img.clone())
        _, h, w = img.shape
        erase_mask = torch.zeros(1, h, w, dtype=torch.float32)
        if self.mask_ratio > 0:
            area = h * w
            for _ in range(10):
                target_area = random.uniform(self.mask_ratio * 0.5, self.mask_ratio) * area
                aspect_ratio = random.uniform(0.3, 3.3)
                eh = int(round((target_area * aspect_ratio) ** 0.5))
                ew = int(round((target_area / aspect_ratio) ** 0.5))
                if 0 < eh < h and 0 < ew < w:
                    y0 = random.randint(0, h - eh)
                    x0 = random.randint(0, w - ew)
                    img[:, y0:y0 + eh, x0:x0 + ew] = 0.0
                    erase_mask[:, y0:y0 + eh, x0:x0 + ew] = 1.0
                    break
        return img, erase_mask

    def __getitem__(self, idx):
        _, stem, _ = self.items[idx]

        clean_global = torch.load(self.image_dir / stem / 'global.pt', weights_only=True)

        # Sample one set of geo params and apply to the global image; the same
        # transform is applied to every local tile so positions stay consistent.
        geo_params = _sample_geo_params(self.aug_flags) if self.augment else (0, 0, 0)
        if self.augment:
            clean_global = _apply_geo_tensor(clean_global, geo_params)

        aug_global, erase_mask = self._augment(clean_global)

        # Randomly sample num_local_crops tiles from this stem (with replacement so
        # small stems are handled gracefully).
        sampled = random.choices(self.stem_to_items[stem], k=self.num_local_crops)

        pos_list = []
        local_target_list = []
        for sp_path, _, sp_entry in sampled:
            local_tile = torch.load(sp_path, weights_only=True)
            H, W = sp_entry['image_H'], sp_entry['image_W']
            y, x = sp_entry['y'], sp_entry['x']
            h, w = sp_entry['height'], sp_entry['width']
            pos = torch.tensor([y / H, x / W, (y + h) / H, (x + w) / W], dtype=torch.float32)

            if self.augment:
                local_tile = _apply_geo_tensor(local_tile, geo_params)
                pos = _apply_geo_pos(pos, geo_params)

            local_target = F.interpolate(
                local_tile.unsqueeze(0),
                size=(self.crop_size, self.crop_size),
                mode='bilinear', align_corners=True,
            ).squeeze(0)

            pos_list.append(pos)
            local_target_list.append(local_target)

        pos_stack = torch.stack(pos_list)             # (N, 4)
        local_target_stack = torch.stack(local_target_list)  # (N, C, cs, cs)

        return aug_global, pos_stack, clean_global, local_target_stack, erase_mask

