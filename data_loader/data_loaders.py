import numpy as np
from collections import defaultdict
from torch.utils.data.sampler import SubsetRandomSampler
from base import BaseDataLoader
from data_loader.datasets import GlobalDataset, LocalDataset, LocalGlobalDataset, SSLGlobalLocalDataset


class GlobalDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.dataset = GlobalDataset(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class LocalDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.dataset = LocalDataset(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class LocalGlobalDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.dataset = LocalGlobalDataset(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        # Group tile indices by stem (patient / global image) so that all tiles
        # from the same slide land entirely in train or entirely in validation.
        stem_to_indices = defaultdict(list)
        for idx, (_, stem, _) in enumerate(self.dataset.items):
            stem_to_indices[stem].append(idx)

        stems = list(stem_to_indices.keys())
        np.random.seed(0)
        np.random.shuffle(stems)

        # split can be a float (fraction of slides) or an int (exact number of val slides)
        n_val = split if isinstance(split, int) else max(1, int(len(stems) * split))

        val_stems   = stems[:n_val]
        train_stems = stems[n_val:]

        train_idx = [i for s in train_stems for i in stem_to_indices[s]]
        val_idx   = [i for s in val_stems   for i in stem_to_indices[s]]

        self.shuffle = False
        self.n_samples = len(train_idx)

        return SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx)


class SSLGlobalLocalDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1,
                 crop_size=32, noise_std=0.1, color_scale_range=0.2,
                 blur_kernel=5, cutout_ratio=0.2,
                 p_noise=0.5, p_color=0.5, p_blur=0.3, p_cutout=0.3):
        self.dataset = SSLGlobalLocalDataset(
            data_dir, crop_size=crop_size,
            noise_std=noise_std, color_scale_range=color_scale_range,
            blur_kernel=blur_kernel, cutout_ratio=cutout_ratio,
            p_noise=p_noise, p_color=p_color, p_blur=p_blur, p_cutout=p_cutout,
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
