"""Train ONE fold of ONE ROI-ablation arm.

Invoked as a subprocess by train_ablation.py; not normally run by hand. It makes
the two changes to the repo's training entry point that a config file cannot
express, without editing any file in the repo:

  1. registers NoiseROIDataLoader into data_loader.data_loaders, so a config can
     name it exactly the way it names LocalGlobalDataLoader;
  2. attaches the ROI weight probe to up5 once the Trainer has built the model.

Both are done by rebinding names train.py has already imported, so train.main()
runs verbatim afterwards. Everything else -- optimiser, schedule, early stopping,
the LOCO split, the post-training held-out test evaluation -- is inherited from
the flagship config unchanged, so the arms differ from it only in the ROI.

The `noise` arm sees more tiles per epoch than `noise_guided` because dropping
roi_exclusion drops the tile filter. That is the experiment, not a confound: the
ROI's contribution includes deciding which tiles exist. Both arms stop on
val_loss convergence and report a val-selected best checkpoint.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
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
    ap = argparse.ArgumentParser(description='one fold of one ROI-ablation arm')
    ap.add_argument('-c', '--config', required=True, type=str)
    ap.add_argument('-r', '--resume', default=None, type=str)
    ap.add_argument('-d', '--device', default=None, type=str)
    ap.add_argument('--probe-every', default=1, type=int)
    ap.add_argument('--probe-hist-every', default=50, type=int)
    known, _ = ap.parse_known_args()

    register()
    _patch_trainer(known.probe_every, known.probe_hist_every)

    # from_args wants the parser itself, not the parsed namespace. The two probe
    # flags are already consumed above and are ignored by ConfigParser.
    config = ConfigParser.from_args(ap, [])
    train_mod.main(config)


if __name__ == '__main__':
    main()
