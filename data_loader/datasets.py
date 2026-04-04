from pathlib import Path
import json
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset


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

        return local, glob, pos, mask, global_mask


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
                 p_channel_drop=0.3):
        data_dir = Path(data_dir)
        self.image_dir = data_dir / 'images'
        self.crop_size = crop_size

        self.transform = T.Compose([
            T.RandomApply([T.Lambda(lambda x: x + torch.randn_like(x) * noise_std)], p=p_noise),
            T.RandomApply([T.Lambda(lambda x: x * (1.0 + (torch.rand(3, 1, 1) * 2 - 1) * color_scale_range))], p=p_color),
            T.RandomApply([T.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0))], p=p_blur),
            T.RandomApply([T.Lambda(lambda x: x.index_fill(0, torch.tensor([random.randint(0, 2)]), 0.0))], p=p_channel_drop),
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
        """Apply random appearance-only augmentations to a (3, H, W) tensor."""
        return self.transform(img.clone())

    def __getitem__(self, idx):
        patch_path, stem, entry = self.items[idx]

        clean_global = torch.load(self.image_dir / stem / 'global.pt', weights_only=True)
        aug_global = self._augment(clean_global)

        H, W = entry['image_H'], entry['image_W']
        y, x = entry['y'], entry['x']
        h, w = entry['height'], entry['width']
        pos = torch.tensor([y / H, x / W, (y + h) / H, (x + w) / W], dtype=torch.float32)

        local_tile = torch.load(patch_path, weights_only=True)   # (3, 256, 256)
        local_target = F.interpolate(
            local_tile.unsqueeze(0),
            size=(self.crop_size, self.crop_size),
            mode='bilinear', align_corners=True,
        ).squeeze(0)   # (3, crop_size, crop_size)

        return aug_global, pos, clean_global, local_target
