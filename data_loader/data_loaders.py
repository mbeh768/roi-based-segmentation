import numpy as np
from collections import defaultdict
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from base import BaseDataLoader
from data_loader.datasets import GlobalDataset, LocalDataset, LocalGlobalDataset, SSLGlobalLocalDataset, BYOLDataset, MAEDataset


class ForegroundBalancedSampler(Sampler):
    """
    Yields indices so every `batch_size` consecutive chunk contains
    `ceil(batch_size * fg_ratio)` foreground tiles and the rest background.
    Pools wrap around (with reshuffle) independently, so the minority pool is
    upsampled as needed.
    """
    def __init__(self, fg_indices, bg_indices, batch_size, fg_ratio):
        self.fg = list(fg_indices)
        self.bg = list(bg_indices)
        self.batch_size = batch_size
        self.fg_per_batch = max(1, int(np.ceil(batch_size * fg_ratio)))
        self.bg_per_batch = batch_size - self.fg_per_batch
        self.n_samples = len(self.fg) + len(self.bg)
        assert len(self.fg) > 0, "No foreground tiles found — cannot enforce fg_ratio"
        assert self.bg_per_batch == 0 or len(self.bg) > 0, "No background tiles found"

    def __iter__(self):
        n_batches = self.n_samples // self.batch_size
        fg_pool = self.fg.copy()
        bg_pool = self.bg.copy()
        np.random.shuffle(fg_pool)
        np.random.shuffle(bg_pool)
        fi = bi = 0
        for _ in range(n_batches):
            batch = []
            for _ in range(self.fg_per_batch):
                if fi >= len(fg_pool):
                    np.random.shuffle(fg_pool)
                    fi = 0
                batch.append(fg_pool[fi]); fi += 1
            for _ in range(self.bg_per_batch):
                if bi >= len(bg_pool):
                    np.random.shuffle(bg_pool)
                    bi = 0
                batch.append(bg_pool[bi]); bi += 1
            np.random.shuffle(batch)
            for idx in batch:
                yield idx

    def __len__(self):
        return (self.n_samples // self.batch_size) * self.batch_size


class GlobalDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.dataset = GlobalDataset(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class LocalDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.dataset = LocalDataset(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class LocalGlobalDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0,
                 num_workers=1, visualize=None,
                 hflip=0.0, vflip=0.0, rot90=0.0,
                 fg_ratio=0.0, fg_min=0.05):
        self.dataset = LocalGlobalDataset(data_dir, hflip=hflip, vflip=vflip, rot90=rot90)
        if visualize:
            self.dataset.save_visualizations(visualize)
        self._batch_size = batch_size
        self.fg_ratio = fg_ratio
        self.fg_min = fg_min
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def _split_sampler(self, split):
        if split == 0.0 and self.fg_ratio == 0.0:
            return None, None

        # Group tile indices by stem (patient / global image) so that all tiles
        # from the same slide land entirely in train or entirely in validation.
        stem_to_indices = defaultdict(list)
        for idx, (_, stem, _) in enumerate(self.dataset.items):
            stem_to_indices[stem].append(idx)

        if split == 0.0:
            train_idx = list(range(len(self.dataset)))
            val_sampler = None
        else:
            stems = list(stem_to_indices.keys())
            np.random.seed(0)
            np.random.shuffle(stems)
            n_val = split if isinstance(split, int) else max(1, int(len(stems) * split))
            val_stems   = stems[:n_val]
            train_stems = stems[n_val:]
            train_idx = [i for s in train_stems for i in stem_to_indices[s]]
            val_idx   = [i for s in val_stems   for i in stem_to_indices[s]]
            val_sampler = SubsetRandomSampler(val_idx)

        if self.fg_ratio > 0:
            fg, _ = self.dataset.foreground_indices(self.fg_min)
            fg_set = set(fg)
            fg_train = [i for i in train_idx if i in fg_set]
            bg_train = [i for i in train_idx if i not in fg_set]
            train_sampler = ForegroundBalancedSampler(
                fg_train, bg_train, self._batch_size, self.fg_ratio,
            )
            self.n_samples = len(train_sampler)
        else:
            train_sampler = SubsetRandomSampler(train_idx)
            self.n_samples = len(train_idx)

        self.shuffle = False
        return train_sampler, val_sampler


class SSLGlobalLocalDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1,
                 crop_size=32, noise_std=0.1, color_scale_range=0.2,
                 blur_kernel=5,
                 p_noise=0.5, p_color=0.5, p_blur=0.3,
                 mask_ratio=0.0, mask_patch_size=16,
                 p_intensity_invert=0.0,
                 hflip=0.0, vflip=0.0, rot90=0.0):
        self.dataset = SSLGlobalLocalDataset(
            data_dir, crop_size=crop_size,
            noise_std=noise_std, color_scale_range=color_scale_range,
            blur_kernel=blur_kernel,
            p_noise=p_noise, p_color=p_color, p_blur=p_blur,
            mask_ratio=mask_ratio, mask_patch_size=mask_patch_size,
            p_intensity_invert=p_intensity_invert,
            hflip=hflip, vflip=vflip, rot90=rot90,
        )
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        stem_to_indices = defaultdict(list)
        for idx, (_, stem, _) in enumerate(self.dataset.items):
            stem_to_indices[stem].append(idx)

        stems = list(stem_to_indices.keys())
        np.random.seed(0)
        np.random.shuffle(stems)

        n_val = split if isinstance(split, int) else max(1, int(len(stems) * split))

        val_stems   = stems[:n_val]
        train_stems = stems[n_val:]

        train_idx = [i for s in train_stems for i in stem_to_indices[s]]
        val_idx   = [i for s in val_stems   for i in stem_to_indices[s]]

        self.shuffle = False
        self.n_samples = len(train_idx)

        return SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx)


class BYOLDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1,
                 noise_std=0.2, color_scale_range=0.3,
                 blur_kernel=5,
                 p_noise=0.5, p_color=0.5, p_blur=0.5,
                 p_intensity_invert=0.2,
                 mask_ratio=0.0, mask_patch_size=16,
                 hflip=0.0, vflip=0.0, rot90=0.0):
        self.dataset = BYOLDataset(
            data_dir,
            noise_std=noise_std, color_scale_range=color_scale_range,
            blur_kernel=blur_kernel,
            p_noise=p_noise, p_color=p_color, p_blur=p_blur,
            p_intensity_invert=p_intensity_invert,
            mask_ratio=mask_ratio, mask_patch_size=mask_patch_size,
            hflip=hflip, vflip=vflip, rot90=rot90,
        )
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        stem_to_indices = defaultdict(list)
        for idx, (_, stem, _) in enumerate(self.dataset.items):
            stem_to_indices[stem].append(idx)

        stems = list(stem_to_indices.keys())
        np.random.seed(0)
        np.random.shuffle(stems)

        n_val = split if isinstance(split, int) else max(1, int(len(stems) * split))
        val_stems   = stems[:n_val]
        train_stems = stems[n_val:]

        train_idx = [i for s in train_stems for i in stem_to_indices[s]]
        val_idx   = [i for s in val_stems   for i in stem_to_indices[s]]

        self.shuffle = False
        self.n_samples = len(train_idx)

        return SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx)


class MAEDataLoader(BaseDataLoader):
    """One item per global (with all its tiles). Split is item-level, so
    stems are already disjoint between train and val by construction."""
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1,
                 hflip=0.0, vflip=0.0, rot90=0.0):
        self.dataset = MAEDataset(data_dir, hflip=hflip, vflip=vflip, rot90=rot90)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        n = len(self.dataset)
        np.random.seed(0)
        indices = np.arange(n)
        np.random.shuffle(indices)

        n_val = split if isinstance(split, int) else max(1, int(n * split))
        val_idx = indices[:n_val].tolist()
        train_idx = indices[n_val:].tolist()

        self.shuffle = False
        self.n_samples = len(train_idx)

        return SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx)
