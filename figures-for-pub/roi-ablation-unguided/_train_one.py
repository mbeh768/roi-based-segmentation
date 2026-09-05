"""Drop-in replacement for train.py: ROI ablation WITHOUT tile selection.

Used as run_kfold.sh's TRAIN_ENTRY, so this arm reuses the same runner,
preflight, skip logic and summary as the baseline and the guided ablation.

This is the third point in the decomposition. The ROI does two separate jobs,
and the three arms separate them:

    arm                     selects tiles   model sees     isolates
    -------------------------------------------------------------------
    baseline                real ROI        real ROI       --
    roi-ablation            real ROI        N(0,1) noise   the input prior
    roi-ablation-unguided   nothing         N(0,1) noise   + tile selection

The gap between the second and third is what tile selection is worth. Without
it the pool goes from 3698 to 19220 tiles, so the model sees every window on the
slide rather than only those overlapping the annotation -- and the discarded
tiles contain no ventricle at all, so this arm trains on a pool that is ~81%
guaranteed-empty background at roughly 5.2x the steps per epoch. That cost is
the experiment, not a confound: choosing which tiles exist is part of what the
ROI contributes.

Two changes are hardcoded rather than exposed, because this folder IS this arm;
a different ablation gets its own folder:

    data_loader.type  -> NoiseROIDataLoader   (the model sees noise)
    roi_exclusion     -> 0                    (no tile selection)

Everything else is inherited from the BASELINE config.json, so lr, batch size,
skip_proj, the LOCO split and the stopping rule cannot drift away from the arms
this is compared against.
"""
import argparse
import collections
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# the repo root is the directory holding train.py, at whatever depth
ROOT = next(p for p in HERE.parents if (p / 'train.py').exists())
sys.path.insert(0, str(ROOT))
# the noise loader and the weight probe live with the guided arm; this arm
# differs from it only in tile selection, so it reuses them rather than copying
sys.path.insert(0, str(HERE.parent / 'roi-ablation'))

from parse_config import ConfigParser      # noqa: E402
import train as train_mod                  # noqa: E402
from noise_roi import register             # noqa: E402
from roi_weight_probe import attach        # noqa: E402


def _patch_trainer(probe_every, probe_hist_every):
    """Subclass Trainer purely to get a hook after the model is built.

    The probe has to attach to a live nn.Module, and train.main() constructs the
    Trainer itself with no callback to hand it to. Rebinding the name train.py
    already imported is the only way in without editing train.py. The subclass
    adds nothing else -- __init__ runs the parent's verbatim, then attaches.
    """
    original = train_mod.Trainer

    class ProbedTrainer(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # DataParallel would hide up5 behind .module; n_gpu is 1 here, but
            # the probe resolves layers by attribute, so unwrap defensively.
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
    ap = argparse.ArgumentParser(description='ROI ablation without tile selection, one fold')
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
        # int, not bool: argparse reads the string "0" as True. 0 is falsy where
        # the dataset tests it, which is what turns tile selection off.
        CustomArgs(['--roi_exclusion'], type=int, target='data_loader;args;roi_exclusion'),
    ]
    # from_args registers every option flag with default=None, so these have to
    # arrive as real arguments rather than defaults.
    if '--loader_type' not in sys.argv:
        sys.argv += ['--loader_type', 'NoiseROIDataLoader']
    if '--roi_exclusion' not in sys.argv:
        sys.argv += ['--roi_exclusion', '0']

    config = ConfigParser.from_args(ap, options)
    train_mod.main(config)


if __name__ == '__main__':
    main()
