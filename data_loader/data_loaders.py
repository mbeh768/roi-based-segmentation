import numpy as np
from collections import defaultdict
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from base import BaseDataLoader
from data_loader.datasets import GlobalDataset, LocalDataset, LocalGlobalDataset


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
                 fg_ratio=0.0, fg_min=0.05,
                 require_roi=False, roi_jitter=0.0,
                 roi_dropout=0.0, roi_distort=0.0, roi_distort_strength=0.3,
                 roi_distort_end=None, roi_distort_ramp_epochs=0,
                 roi_dropout_end=None, roi_dropout_ramp_epochs=0,
                 roi_exclusion=False, roi_exclusion_radius=0.05,
                 split_seed=0, n_folds=0, fold=0):
        self.dataset = LocalGlobalDataset(
            data_dir, hflip=hflip, vflip=vflip, rot90=rot90,
            require_roi=require_roi, roi_jitter=roi_jitter,
            roi_dropout=roi_dropout, roi_distort=roi_distort,
            roi_distort_strength=roi_distort_strength,
            roi_exclusion=roi_exclusion, roi_exclusion_radius=roi_exclusion_radius,
        )
        if visualize:
            self.dataset.save_visualizations(visualize)
        self._batch_size = batch_size
        self.fg_ratio = fg_ratio
        self.fg_min = fg_min
        self.split_seed = split_seed
        self.n_folds = n_folds
        self.fold = fold
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
            # split_seed controls which slides land in validation. Vary it across
            # runs to check the reported metric isn't an artifact of one lucky split.
            np.random.RandomState(self.split_seed).shuffle(stems)
            if self.n_folds and self.n_folds >= 2:
                # k-fold: partition the (seed-shuffled) slides into n_folds chunks and
                # hold out chunk `fold` for validation. Keep split_seed FIXED across the
                # k runs and vary only `fold` (0..n_folds-1) so each slide is validated
                # exactly once and weight init stays constant — isolating the split
                # effect. validation_split is ignored in fold mode (val size = 1/k).
                chunks = [list(c) for c in np.array_split(np.array(stems, dtype=object), self.n_folds)]
                f = self.fold % self.n_folds
                val_stems   = chunks[f]
                train_stems = [s for i, c in enumerate(chunks) if i != f for s in c]
            else:
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

