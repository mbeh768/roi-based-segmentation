import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
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


# Some slides are two physical halves of the same tissue sample (cut from the
# same block and stained together — see verification/gather_evidence.py's
# PAIRS) and must never land on opposite sides of a train/val split: validating
# on one half while training on its twin leaks adjacent-tissue signal. Maps
# stem -> case id; a stem absent from this dict is its own (singleton) case.
CASE_GROUPS = {
    '21PControl-9-1of2':   '21PControl-9',
    '21PControl-9-2of2':   '21PControl-9',
    '21PControl-11-1of2':  '21PControl-11',
    '21PControl-11-2of2':  '21PControl-11',
    'AJK056-10-2of2':      'AJK056-10',
    'AJKV056-10-1-of-2':   'AJK056-10',   # sic: "AJK" vs "AJKV" typo in the source filename
    'AJKY121-8-1':         'AJKY121-8',
    'AJKY121-8-2':         'AJKY121-8',
    'AJKY121-9-1':         'AJKY121-9',
    'AJKY121-9-2':         'AJKY121-9',
    'FL_22P003-6_1of2':    'FL_22P003-6',
    'FL_22P003-6_2of2':    'FL_22P003-6',
}


def case_id(stem):
    return CASE_GROUPS.get(stem, stem)


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
                 num_workers=1,
                 hflip=0.0, vflip=0.0, rot90=0.0,
                 fg_ratio=0.0, fg_min=0.05,
                 require_roi=False, roi_jitter=0.0,
                 roi_dropout=0.0, roi_distort=0.0, roi_distort_strength=0.3,
                 roi_distort_end=None, roi_distort_ramp_epochs=0,
                 roi_dropout_end=None, roi_dropout_ramp_epochs=0,
                 roi_exclusion=False, roi_exclusion_radius=0.05,
                 split_seed=0, n_folds=0, fold=0, n_val_cases=0):
        self.dataset = self._make_dataset(
            data_dir, hflip=hflip, vflip=vflip, rot90=rot90,
            require_roi=require_roi, roi_jitter=roi_jitter,
            roi_dropout=roi_dropout, roi_distort=roi_distort,
            roi_distort_strength=roi_distort_strength,
            roi_exclusion=roi_exclusion, roi_exclusion_radius=roi_exclusion_radius,
        )
        self._batch_size = batch_size
        self.fg_ratio = fg_ratio
        self.fg_min = fg_min
        self.split_seed = split_seed
        self.n_folds = n_folds
        self.fold = fold
        self.n_val_cases = n_val_cases
        self.test_indices = []          # filled by _split_sampler in fold mode
        self.split_cases = {}           # {'train': [...], 'val': [...], 'test': [...]}
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def _split_sampler(self, split):
        if split == 0.0 and self.fg_ratio == 0.0:
            return None, None

        # Group tile indices by case (CASE_GROUPS-merged stem) so that all tiles
        # from the same slide -- and from its paired twin half, if it has one --
        # land entirely in train or entirely in validation.
        case_to_indices = defaultdict(list)
        for idx, (_, stem, _) in enumerate(self.dataset.items):
            case_to_indices[case_id(stem)].append(idx)

        if split == 0.0:
            train_idx = list(range(len(self.dataset)))
            val_sampler = None
        else:
            cases = list(case_to_indices.keys())
            # split_seed controls which cases land in validation. Vary it across
            # runs to check the reported metric isn't an artifact of one lucky split.
            np.random.RandomState(self.split_seed).shuffle(cases)
            if self.n_folds and self.n_folds >= 2:
                # k-fold: partition the (seed-shuffled) cases into n_folds chunks and
                # hold out chunk `fold` as the TEST set. Keep split_seed FIXED across
                # the k runs and vary only `fold` (0..n_folds-1) so each case is tested
                # exactly once and weight init stays constant — isolating the split
                # effect. validation_split is ignored in fold mode.
                # n_folds == len(cases) gives leave-one-case-out (the smallest possible
                # test set per fold); n_folds > len(cases) would leave some folds
                # empty, so it's rejected rather than silently under-filled.
                if self.n_folds > len(cases):
                    raise ValueError(
                        f"n_folds={self.n_folds} exceeds the number of cases "
                        f"({len(cases)}) after pair-grouping -- can't leave one "
                        f"case out per fold.")
                chunks = [list(c) for c in np.array_split(np.array(cases, dtype=object), self.n_folds)]
                f = self.fold % self.n_folds
                test_cases = chunks[f]

                # Carve the validation set out of the REMAINING cases, never out of
                # the test case: early stopping and best-checkpoint selection read
                # val_loss, so anything in val has effectively been seen by model
                # selection and can no longer serve as a held-out estimate. With
                # n_val_cases=0 this degenerates to the old two-way split, where the
                # reported metric is optimistically biased by that selection.
                rest = [c for c in cases if c not in set(test_cases)]
                k = min(self.n_val_cases, max(len(rest) - 1, 0))
                if self.n_val_cases and k < self.n_val_cases:
                    raise ValueError(
                        f"n_val_cases={self.n_val_cases} leaves no training cases "
                        f"(only {len(rest)} non-test cases available)")
                # Rotate the val window with the fold so the same slides aren't
                # always the ones spent on model selection.
                if k > 0:
                    start = (self.fold * k) % len(rest)
                    val_cases = [rest[(start + i) % len(rest)] for i in range(k)]
                else:
                    val_cases = list(test_cases)      # legacy two-way behaviour
                val_set = set(val_cases)
                train_cases = [c for c in rest if c not in val_set]

                self.test_indices = [i for c in test_cases for i in case_to_indices[c]]
                self.split_cases = {'train': list(train_cases),
                                    'val': list(val_cases),
                                    'test': list(test_cases)}
            else:
                n_val = split if isinstance(split, int) else max(1, int(len(cases) * split))
                val_cases   = cases[:n_val]
                train_cases = cases[n_val:]
            train_idx = [i for c in train_cases for i in case_to_indices[c]]
            val_idx   = [i for c in val_cases   for i in case_to_indices[c]]
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

    def _make_dataset(self, data_dir, **kw):
        """Overridden by subclasses that need a different Dataset class."""
        return LocalGlobalDataset(data_dir, **kw)

    def split_test(self):
        """Loader over the fold's test cases -- never trained on and never read by
        early stopping, so it is only ever touched once, after training, by
        whichever checkpoint model selection already settled on."""
        if not self.test_indices:
            return None
        return DataLoader(sampler=SubsetRandomSampler(self.test_indices),
                          **self.init_kwargs)