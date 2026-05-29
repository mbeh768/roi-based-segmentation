import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    self._save_checkpoint(epoch, save_best=True)
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=False)

        self._save_epoch_plots()

    def _save_epoch_plots(self):
        """Parse info.log for per-epoch metrics and save one PNG per base metric.
        Train and val lines of the same metric share a graph. Y-limits: loss [0, 1.5], iou [0, 1]."""
        import re
        from pathlib import Path
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning('matplotlib not installed; skipping epoch plots')
            return

        log_dir = Path(self.config.log_dir)
        log_path = log_dir / 'info.log'
        if not log_path.exists():
            self.logger.warning(f'info.log not found at {log_path}; skipping epoch plots')
            return

        # block parser: lines like "... - INFO -     key            : value"
        kv_re = re.compile(r'-\s*INFO\s*-\s*(\S+)\s*:\s*([-+0-9.eE]+)\s*$')
        epochs = []             # list of {key: float}
        current = None
        with open(log_path) as f:
            for line in f:
                m = kv_re.search(line)
                if not m:
                    continue
                key, val = m.group(1), m.group(2)
                try:
                    val = float(val)
                except ValueError:
                    continue
                if key == 'epoch':
                    current = {'epoch': int(val)}
                    epochs.append(current)
                elif current is not None:
                    current[key] = val

        if not epochs:
            self.logger.warning('no epoch blocks parsed from info.log; skipping plots')
            return

        # discover base metrics: any key seen (without val_ prefix)
        keys = set()
        for e in epochs:
            keys.update(e.keys())
        keys.discard('epoch')
        bases = {k[4:] if k.startswith('val_') else k for k in keys}

        ylim_for = lambda name: (
            (0, 1.5) if 'loss' in name else
            (0, 1) if 'iou' in name else
            None
        )

        for base in sorted(bases):
            xs_train, ys_train = [], []
            xs_val, ys_val = [], []
            for e in epochs:
                ep = e['epoch']
                if base in e:
                    xs_train.append(ep); ys_train.append(e[base])
                if f'val_{base}' in e:
                    xs_val.append(ep); ys_val.append(e[f'val_{base}'])
            if not xs_train and not xs_val:
                continue

            fig, ax = plt.subplots(figsize=(8, 4.5))
            if xs_train:
                ax.plot(xs_train, ys_train, label='train', color='tab:blue')
            if xs_val:
                ax.plot(xs_val, ys_val, label='val', color='tab:orange')
            ax.set_xlabel('epoch')
            ax.set_ylabel(base)
            ax.set_title(base)
            lim = ylim_for(base)
            if lim is not None:
                ax.set_ylim(*lim)
            ax.grid(True, alpha=0.3)
            ax.legend()
            out = log_dir / f'{base}.png'
            fig.tight_layout()
            fig.savefig(out, dpi=100)
            plt.close(fig)
            self.logger.info(f'saved plot: {out}')

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth (epoch {}) ...".format(epoch))
        else:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
