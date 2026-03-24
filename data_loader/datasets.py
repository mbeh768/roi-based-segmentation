from pathlib import Path
import json
import torch
from torch.utils.data import Dataset


class GlobalVentricleDataset(Dataset):
    """
    Loads only the downsampled global.pt image + corresponding global mask.

    Each patient directory must contain:
        data/images/<stem>/global.pt  — (3, H, W) float32
        data/masks/<stem>/global.pt   — (1, H, W) float32

    Returns (image, mask) — a 2-tuple so the trainer unpacks it as
    (*inputs=[image], target=mask) and calls model(image).
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


class LocalVentricleDataset(Dataset):
    """
    Loads local patch tiles + corresponding masks only — no global context.

    Builds a flat list of all patch_*.pt files across every patient via
    coordinates.json, identical to VentricleDataset but returns only
    (patch_image, patch_mask) so it feeds directly into SimpleUNet.

    Returns (image, mask) — same 2-tuple contract as GlobalVentricleDataset.
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


class VentricleDataset(Dataset):
    """
    Loads local tile + corresponding global image + mask.

    Index is built by reading coordinates.json files written by preprocess.py.
    All files for a given image live together in data_dir/<subdir>/<stem>/:
        global.pt            — (C, H, W) float32  downsampled whole-slide
        patch_YYYY_XXXX.pt   — (C, H, W) float32  local tile
        coordinates.json     — patch index with stem, x, y, image_W, image_H

    Returns (local_tensor, global_tensor, mask_tensor).
    """
    def __init__(self, data_dir):
        data_dir = Path(data_dir)
        self.image_dir = data_dir / 'images'
        self.mask_dir  = data_dir / 'masks'

        # Build flat list of (patch_path, stem, entry) from all coordinates.json files
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

        # normalised tile bounding box [y0, x0, y1, x1] within the tiled image
        H, W = entry['image_H'], entry['image_W']
        y, x = entry['y'], entry['x']
        h, w = entry['height'], entry['width']
        pos = torch.tensor([y / H, x / W, (y + h) / H, (x + w) / W], dtype=torch.float32)

        mask_path = self.mask_dir / stem / patch_path.name
        if mask_path.exists():
            mask = torch.load(mask_path, weights_only=True)
        else:
            mask = torch.zeros(1, local.shape[-2], local.shape[-1])

        return local, glob, pos, mask
