"""Noise-ROI dataset and loader for the ROI ablation arms.

Self-contained: nothing in data_loader/ is modified. These subclass the repo's
LocalGlobalDataset / LocalGlobalDataLoader and change exactly one thing -- the
ROI tensor handed to the model is replaced with Gaussian noise.

THE TWO ARMS the paper reports differ only in `roi_exclusion`, because the ROI
does two separate jobs in this pipeline and they need separating:

  input prior      a 1-channel ROI crop concatenated onto the 16x16 bottleneck
                   (model/model.py, `x = torch.cat([x, roi_at(...)])`), consumed
                   by up5. This is the only place the network sees the ROI.
  tile selection   with roi_exclusion=True the dataset drops every tile that
                   does not touch the ROI, so the ROI also decides WHICH tiles
                   exist at all. Nothing about this reaches the network as input.

  arm             roi_noise  roi_exclusion   what is removed
  ------------------------------------------------------------------
  noise           True       False           both jobs
  noise_guided    True       True            the input prior only

Comparing the two isolates how much of the ROI's contribution is guidance to
the network versus restriction of the field of view.

NOISE IS REDRAWN ON EVERY ACCESS (unless noise_seed is set). A fixed per-slide
noise field would be a constant the network could learn to recognise as a slide
identifier, which would leak exactly the localisation the ablation is meant to
destroy. Fresh draws carry no information across epochs.

Noise is raw N(0, 1) -- it is not matched to the ROI's binary {0,1} statistics.
The claim being tested is "the channel carries no usable localisation", and
the architecture is unchanged, so parameter count and every code path are
identical to the champion.

Substitution is applied AFTER the parent's __getitem__ rather than by
reimplementing it. The parent still loads and geometrically augments the real
ROI and the result is discarded; that costs one small tensor load per item but
guarantees this class cannot drift from the champion's data path as that path
changes.
"""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader.datasets import LocalGlobalDataset          # noqa: E402
from data_loader.data_loaders import LocalGlobalDataLoader    # noqa: E402


class NoiseROIDataset(LocalGlobalDataset):
    """LocalGlobalDataset whose ROI channel is Gaussian noise.

    noise_seed
        None  -- a fresh draw every access, from the ambient RNG. Correct for
                 training: the noise must not be learnable.
        int   -- draw is a deterministic function of (noise_seed, idx), so an
                 evaluation pass is exactly reproducible and two passes with
                 different seeds give genuinely different noise for the same
                 tiles. The redraw-invariance probe depends on this.
    """

    def __init__(self, *args, noise_seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_seed = noise_seed

    def _noise_like(self, roi, idx):
        if self.noise_seed is None:
            return torch.randn_like(roi)
        g = torch.Generator().manual_seed(int(self.noise_seed) * 1_000_003 + int(idx))
        return torch.randn(roi.shape, generator=g, dtype=roi.dtype)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        if not self.require_roi:
            raise ValueError(
                "NoiseROIDataset needs require_roi=True: the ablation replaces the "
                "ROI channel, it does not remove it. Removing the channel is a "
                "different (architecture-changing) experiment."
            )
        local, glob, pos, roi, mask, global_mask = out
        return local, glob, pos, self._noise_like(roi, idx), mask, global_mask


class NoiseROIDataLoader(LocalGlobalDataLoader):
    """LocalGlobalDataLoader that builds a NoiseROIDataset.

    Accepts every argument the parent does, plus noise_seed. Splitting, folds,
    and the ROI-exclusion tile filter are all inherited unchanged -- in
    particular roi_exclusion still uses the REAL on-file ROI to choose tiles,
    which is what makes the `noise_guided` arm meaningful.
    """

    def __init__(self, *args, noise_seed=None, **kwargs):
        self._noise_seed = noise_seed
        super().__init__(*args, **kwargs)

    def _make_dataset(self, data_dir, **kw):
        return NoiseROIDataset(data_dir, noise_seed=self._noise_seed, **kw)


def register():
    """Make these classes visible to train.py.

    train.py resolves the loader with getattr(data_loader.data_loaders, cfg type),
    so injecting the names into that module lets a config name them without any
    file in data_loader/ being edited.
    """
    import data_loader.data_loaders as module_data
    module_data.NoiseROIDataset = NoiseROIDataset
    module_data.NoiseROIDataLoader = NoiseROIDataLoader
    return module_data
