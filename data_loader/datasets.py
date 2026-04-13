from pathlib import Path
import json
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
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
    Loads local patch + corresponding global image + mask.

    Index is built by reading coordinates.json files written by preprocess.py.
    All files for a given image live together in data_dir/<subdir>/<stem>/:
        global.pt            — (C, H, W) float32  downsampled whole-slide
        patch_YYYY_XXXX.pt   — (C, H, W) float32  local tile
        coordinates.json     — patch index with stem, x, y, image_W, image_H

    Returns (local, glob, pos, mask) where pos = [y0, x0, y1, x1] normalised.
    """
    def __init__(self, data_dir, hflip=0.0, vflip=0.0, rot90=0.0):
        data_dir = Path(data_dir)
        self.image_dir = data_dir / 'images'
        self.mask_dir  = data_dir / 'masks'
        self.aug_flags = {'hflip': hflip, 'vflip': vflip, 'rot90': rot90}
        self.augment = any(v > 0 for v in self.aug_flags.values())

        # Photometric — applied to local only (add blur, erasure, etc. here)
        self.photo_transform = None

        self.items = []
        for coords_file in sorted(self.image_dir.glob('*/coordinates.json')):
            stem = coords_file.parent.name
            with open(coords_file) as f:
                entries = json.load(f)
            for entry in entries:
                patch_path = coords_file.parent / entry['filename']
                self.items.append((patch_path, stem, entry))

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

        if self.augment:
            local, glob, mask, global_mask, pos = _geo_augment(
                local, glob, mask, global_mask, pos=pos, flags=self.aug_flags,
            )
        if self.photo_transform is not None:
            local = self.photo_transform(local)

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

        n = min(n_samples, len(self.items))
        idxs = random.sample(range(len(self.items)), n)
        for i in idxs:
            # before
            local_b, glob_b, pos_b, mask_b, gmask_b = self[i]
            # after
            self.augment = True
            local_a, glob_a, pos_a, mask_a, gmask_a = self[i]
            self.augment = False

            border_px = 4
            border_color = (0, 255, 0)
            rows = []
            for local, glob, pos, mask, gmask, label in [
                (local_b, glob_b, pos_b, mask_b, gmask_b, 'before'),
                (local_a, glob_a, pos_a, mask_a, gmask_a, 'after'),
            ]:
                g = _draw_pos(_tensor_to_bgr(glob),  pos)
                gm = _draw_pos(_tensor_to_bgr(gmask), pos)
                l = _tensor_to_bgr(local)
                m = _tensor_to_bgr(mask)
                # resize local/mask to match global height
                gh = g.shape[0]
                l = cv2.resize(l, (gh, gh))
                m = cv2.resize(m, (gh, gh))
                panels = [
                    cv2.copyMakeBorder(p, border_px, border_px, border_px, border_px,
                                       cv2.BORDER_CONSTANT, value=border_color)
                    for p in (g, gm, l, m)
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

    Returns (aug_global, pos, clean_global, local_target) where:
        aug_global:   (3, H, W) augmented global image (model input)
        pos:          (4,) normalised tile region [y0, x0, y1, x1]
        clean_global: (3, H, W) original global image (global reconstruction target)
        local_target: (3, crop_size, crop_size) downsampled local tile (local reconstruction target)

    No masks needed -- works with --nomask preprocessed data.
    """
    def __init__(self, data_dir, crop_size=32,
                 noise_std=0.1, color_scale_range=0.2,
                 blur_kernel=5,
                 p_noise=0.5, p_color=0.5, p_blur=0.3,
                 mask_ratio=0.0, mask_patch_size=16,
                 p_intensity_invert=0.0,
                 hflip=0.0, vflip=0.0, rot90=0.0):
        data_dir = Path(data_dir)
        self.image_dir = data_dir / 'images'
        self.crop_size = crop_size
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
        for coords_file in sorted(self.image_dir.glob('*/coordinates.json')):
            stem = coords_file.parent.name
            with open(coords_file) as f:
                entries = json.load(f)
            for entry in entries:
                patch_path = coords_file.parent / entry['filename']
                self.items.append((patch_path, stem, entry))

        assert len(self.items) > 0, f"No patches found via coordinates.json in {self.image_dir}"

    def __len__(self):
        return len(self.items)

    def _augment(self, img):
        """Apply random augmentations to a (C, H, W) tensor."""
        img = self.transform(img.clone())
        if self.mask_ratio > 0:
            img = T.RandomErasing(
                p=1.0, scale=(self.mask_ratio * 0.5, self.mask_ratio),
                ratio=(0.3, 3.3), value=0
            )(img)
        return img

    def __getitem__(self, idx):
        patch_path, stem, entry = self.items[idx]

        clean_global = torch.load(self.image_dir / stem / 'global.pt', weights_only=True)
        local_tile = torch.load(patch_path, weights_only=True)   # (3, 256, 256)

        H, W = entry['image_H'], entry['image_W']
        y, x = entry['y'], entry['x']
        h, w = entry['height'], entry['width']
        pos = torch.tensor([y / H, x / W, (y + h) / H, (x + w) / W], dtype=torch.float32)

        if self.augment:
            clean_global, local_tile, pos = _geo_augment(
                clean_global, local_tile, pos=pos, flags=self.aug_flags,
            )

        aug_global = self._augment(clean_global)

        local_target = F.interpolate(
            local_tile.unsqueeze(0),
            size=(self.crop_size, self.crop_size),
            mode='bilinear', align_corners=True,
        ).squeeze(0)   # (3, crop_size, crop_size)

        return aug_global, pos, clean_global, local_target


class BYOLDataset(Dataset):
    """
    Returns two photometrically- and geometrically-augmented views of a global
    image, the full local tile as the auxiliary reconstruction target, the
    tile's normalized position in the global, and per-view geometric aug
    params so the model can unwarp features back to canonical frame before
    dense BYOL loss.

    Each __getitem__ yields (v1_global, v2_global, local_target, pos, g1, g2).
    g1 and g2 are int64 tensors of shape (3,) = (hflip, vflip, rot90_k).
    pos and local_target are transformed with g1 so they align with v1's frame
    (the view the local_head decodes from).
    """
    def __init__(self, data_dir,
                 noise_std=0.2, color_scale_range=0.3,
                 blur_kernel=5,
                 p_noise=0.5, p_color=0.5, p_blur=0.5,
                 p_intensity_invert=0.2,
                 mask_ratio=0.0, mask_patch_size=16,
                 hflip=0.0, vflip=0.0, rot90=0.0):
        data_dir = Path(data_dir)
        self.image_dir = data_dir / 'images'
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size
        self.aug_flags = {'hflip': hflip, 'vflip': vflip, 'rot90': rot90}
        self.augment = any(v > 0 for v in self.aug_flags.values())

        self.photo_transform = T.Compose([
            T.RandomApply([T.Lambda(lambda x: x + torch.randn_like(x) * noise_std)], p=p_noise),
            T.RandomApply([T.Lambda(lambda x: x * (1.0 + (torch.rand(x.shape[0], 1, 1) * 2 - 1) * color_scale_range))], p=p_color),
            T.RandomApply([T.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0))], p=p_blur),
            T.RandomApply([T.Lambda(lambda x: -x)], p=p_intensity_invert),
        ])

    def _sparse_mask(self, x):
        """MAE-style patch masking: zero out mask_ratio fraction of
        mask_patch_size x mask_patch_size patches."""
        if self.mask_ratio <= 0:
            return x
        _, H, W = x.shape
        ps = self.mask_patch_size
        gh, gw = H // ps, W // ps
        n_patches = gh * gw
        n_mask = int(round(n_patches * self.mask_ratio))
        if n_mask == 0:
            return x
        idx = torch.randperm(n_patches)[:n_mask]
        patch_mask = torch.ones(n_patches, dtype=x.dtype)
        patch_mask[idx] = 0
        patch_mask = patch_mask.view(1, gh, gw)
        pixel_mask = patch_mask.repeat_interleave(ps, dim=-2).repeat_interleave(ps, dim=-1)
        return x * pixel_mask

        self.items = []
        for coords_file in sorted(self.image_dir.glob('*/coordinates.json')):
            stem = coords_file.parent.name
            with open(coords_file) as f:
                entries = json.load(f)
            for entry in entries:
                patch_path = coords_file.parent / entry['filename']
                self.items.append((patch_path, stem, entry))
        assert len(self.items) > 0, f"No patches found via coordinates.json in {self.image_dir}"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        patch_path, stem, entry = self.items[idx]
        glob = torch.load(self.image_dir / stem / 'global.pt', weights_only=True)
        local = torch.load(patch_path, weights_only=True)

        H, W = entry['image_H'], entry['image_W']
        y, x = entry['y'], entry['x']
        h, w = entry['height'], entry['width']
        pos = torch.tensor([y / H, x / W, (y + h) / H, (x + w) / W], dtype=torch.float32)

        if self.augment:
            g1 = _sample_geo_params(self.aug_flags)
            g2 = _sample_geo_params(self.aug_flags)
        else:
            g1 = g2 = (0, 0, 0)

        v1 = _apply_geo_tensor(glob.clone(), g1)
        v2 = _apply_geo_tensor(glob.clone(), g2)
        local_target = _apply_geo_tensor(local, g1)
        pos = _apply_geo_pos(pos, g1)

        v1 = self.photo_transform(v1)
        v2 = self.photo_transform(v2)
        v1 = self._sparse_mask(v1)
        v2 = self._sparse_mask(v2)

        g1_t = torch.tensor(g1, dtype=torch.long)
        g2_t = torch.tensor(g2, dtype=torch.long)
        return v1, v2, local_target, pos, g1_t, g2_t
