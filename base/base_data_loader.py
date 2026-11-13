# Adapted from victoresque/pytorch-template (MIT License).
# See LICENSES/pytorch-template-LICENSE.

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        # Worker processes are torn down and respawned at every epoch boundary by
        # default, and with an epoch this short (~90 batches) that startup is a
        # measurable slice of wall-time -- paid twice over, since the validation
        # loader below respawns its own pool too. Keeping them alive removes it.
        # Only valid when workers exist; num_workers=0 must not carry the flag.
        if num_workers > 0:
            self.init_kwargs['persistent_workers'] = True
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        # Override the template's index-level split: we want source-level split
        # logic, so subclasses must supply it.
        raise NotImplementedError(
            f"{type(self).__name__} must override _split_sampler"
        )

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
