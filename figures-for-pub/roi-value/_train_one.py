"""Drop-in replacement for train.py that runs the ROI ablation.

Used as run_kfold.sh's TRAIN_ENTRY, so the ablation reuses the same runner,
preflight, skip logic and summary as the baseline k-fold. It accepts exactly
train.py's arguments and makes the two changes a config file cannot express,
without editing anything in the repo:

  1. registers NoiseROIDataLoader into data_loader.data_loaders, so the config
     can name it the way it names LocalGlobalDataLoader;
  2. attaches the ROI weight probe to up5 once the Trainer has built the model.

Both are done by rebinding names train.py has already imported, so train.main()
runs verbatim afterwards. Optimiser, schedule, early stopping, the LOCO split
and the held-out test evaluation are all inherited unchanged, so the arm differs
from its baseline only in the ROI the model is shown.
"""
import argparse
import collections
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# the repo root is the directory holding train.py, at whatever depth
ROOT = next(p for p in HERE.parents if (p / 'train.py').exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from parse_config import ConfigParser      # noqa: E402
import train as train_mod                  # noqa: E402
from noise_roi import register             # noqa: E402
from roi_weight_probe import attach        # noqa: E402


def _patch_trainer(probe_every, probe_hist_every):
    original = train_mod.Trainer

    class ProbedTrainer(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # DataParallel would hide up5 behind .module; n_gpu is 1 here, but the
            # probe resolves layers by attribute so unwrap defensively.
            core = getattr(self.model, 'module', self.model)
            if getattr(core, 'use_roi', False):
                self.roi_probe = attach(core, self.writer,
                                        every=probe_every, hist_every=probe_hist_every)
                self.logger.info('ROI weight probe on up5: channel %d of %d, '
                                 'logging every %d forward(s)',
                                 self.roi_probe.roi_start, core.up5.in_channels,
                                 probe_every)

    train_mod.Trainer = ProbedTrainer


def main():
    ap = argparse.ArgumentParser(description='ROI ablation, one fold')
    ap.add_argument('-c', '--config', default='config.json', type=str)
    ap.add_argument('-r', '--resume', default=None, type=str)
    ap.add_argument('-d', '--device', default=None, type=str)
    ap.add_argument('--probe-every', default=1, type=int)
    ap.add_argument('--probe-hist-every', default=50, type=int)
    known, _ = ap.parse_known_args()

    register()
    _patch_trainer(known.probe_every, known.probe_hist_every)

    # MUST mirror train.py's CustomArgs. ConfigParser only applies overrides it
    # is told about, so passing [] here would make --fold, --n_folds and --name
    # silently no-ops: every fold would train fold 0 and write to the same run
    # directory, with nothing in the logs to say so.
    #
    # --loader_type is the extra one, and it is what keeps this arm tied to its
    # baseline: it takes the BASELINE config.json and swaps only the loader
    # class, so the two cannot drift apart when lr, batch size, skip_proj or
    # anything else is retuned.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--seed'], type=int, target='data_loader;args;split_seed'),
        CustomArgs(['--fold'], type=int, target='data_loader;args;fold'),
        CustomArgs(['--n_folds'], type=int, target='data_loader;args;n_folds'),
        CustomArgs(['--data_dir'], type=str, target='data_loader;args;data_dir'),
        CustomArgs(['--name'], type=str, target='name'),
        CustomArgs(['--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--early_stop'], type=int, target='trainer;early_stop'),
        CustomArgs(['--loader_type'], type=str, target='data_loader;type'),
    ]
    # from_args registers every option flag with default=None, so the swap has
    # to arrive as a real argument rather than a default.
    if '--loader_type' not in sys.argv:
        sys.argv += ['--loader_type', 'NoiseROIDataLoader']

    config = ConfigParser.from_args(ap, options)
    train_mod.main(config)


if __name__ == '__main__':
    main()
