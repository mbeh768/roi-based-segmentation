from pathlib import Path
import torch
from torch.utils.data import Dataset


class VentricleDataset(Dataset):
    """
    Loads paired image/mask .pt tensors produced by preprocess.py.

    Expects:
        data_dir/images/<stem>.pt  — (C, H, W) float32 image tensor
        data_dir/masks/<stem>.pt   — (1, H, W) float32 mask tensor  (optional)

    If a mask file is missing, a zero tensor matching the image spatial dims is returned.
    """
    def __init__(self, data_dir):
        self.image_dir = Path(data_dir) / 'images'
        self.mask_dir = Path(data_dir) / 'masks'
        self.image_paths = sorted(self.image_dir.glob('*.pt'))
        assert len(self.image_paths) > 0, f"No .pt files found in {self.image_dir}"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = torch.load(image_path, weights_only=True)

        mask_path = self.mask_dir / image_path.name
        if mask_path.exists():
            mask = torch.load(mask_path, weights_only=True)
        else:
            mask = torch.zeros(1, image.shape[-2], image.shape[-1])

        return image, mask
