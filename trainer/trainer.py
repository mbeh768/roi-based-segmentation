import math
from pathlib import Path

import numpy as np
import cv2
import torch
from base import BaseTrainer
import torch.nn.functional as F

from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.aux_loss_weight = config['trainer'].get('aux_loss_weight', 0.0)

        switch_cfg = config['trainer'].get('loss_switch', None)
        self._switch_cfg  = switch_cfg   # {trigger_metric, threshold, loss} or None
        self._switch_done = False

        dl_args = config['data_loader']['args']
        self._roi_distort_start    = dl_args.get('roi_distort_strength', 0.0)
        self._roi_distort_end      = dl_args.get('roi_distort_end', self._roi_distort_start)
        self._roi_distort_ramp_ep  = dl_args.get('roi_distort_ramp_epochs', 0)

        self._roi_dropout_start    = dl_args.get('roi_dropout', 0.0)
        self._roi_dropout_end      = dl_args.get('roi_dropout_end', self._roi_dropout_start)
        self._roi_dropout_ramp_ep  = dl_args.get('roi_dropout_ramp_epochs', 0)

        train_keys = ['loss', 'main_loss'] if self.aux_loss_weight > 0 else ['loss']
        self.train_metrics = MetricTracker(*train_keys, *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self._save_train_val_previews()

    def _save_train_val_previews(self):
        """Dump each unique stem's global.pt to log_dir/sets/{train,val}/<stem>.png
        so it's possible to visually verify which slides are in each split."""
        out_root = Path(self.config.log_dir) / 'sets'

        def _indices(sampler, dataset_len):
            if sampler is None:
                return list(range(dataset_len))
            if hasattr(sampler, 'fg') and hasattr(sampler, 'bg'):
                return list(sampler.fg) + list(sampler.bg)
            if hasattr(sampler, 'indices'):
                return list(sampler.indices)
            return list(iter(sampler))

        def _stems(dataset, indices):
            items = getattr(dataset, 'items', None)
            if items is not None:
                return {items[i][1] for i in indices}
            pairs = getattr(dataset, 'pairs', None)
            if pairs is not None:
                return {Path(pairs[i][0]).parent.name for i in indices}
            return set()

        def _image_dir(dataset):
            if hasattr(dataset, 'image_dir'):
                return dataset.image_dir
            pairs = getattr(dataset, 'pairs', None)
            if pairs:
                return Path(pairs[0][0]).parent.parent
            return None

        def _dump(stems, dataset, out_dir):
            image_dir = _image_dir(dataset)
            if image_dir is None or not stems:
                return 0
            out_dir.mkdir(parents=True, exist_ok=True)
            n = 0
            for stem in sorted(stems):
                global_path = image_dir / stem / 'global.pt'
                if not global_path.exists():
                    continue
                t = torch.load(global_path, weights_only=True).detach().cpu().float()
                lo, hi = t.min(), t.max()
                t = (t - lo) / (hi - lo + 1e-8)
                arr = (t.numpy() * 255).astype(np.uint8)
                if arr.ndim == 3:
                    if arr.shape[0] == 1:
                        arr = np.repeat(arr, 3, axis=0)
                    arr = arr.transpose(1, 2, 0)
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_dir / f'{stem}.png'), arr)
                n += 1
            return n

        train_ds = self.data_loader.dataset
        train_idx = _indices(self.data_loader.sampler, len(train_ds))
        n_train = _dump(_stems(train_ds, train_idx), train_ds, out_root / 'train')
        self.logger.info(f'Saved {n_train} train globals to {out_root / "train"}')

        if self.do_validation:
            val_ds = self.valid_data_loader.dataset
            val_idx = _indices(self.valid_data_loader.sampler, len(val_ds))
            n_val = _dump(_stems(val_ds, val_idx), val_ds, out_root / 'val')
            self.logger.info(f'Saved {n_val} val globals to {out_root / "val"}')

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        if self._roi_distort_ramp_ep > 0:
            t = min((epoch - 1) / self._roi_distort_ramp_ep, 1.0)
            self.data_loader.dataset.roi_distort_strength = (
                self._roi_distort_start + t * (self._roi_distort_end - self._roi_distort_start)
            )

        if self._roi_dropout_ramp_ep > 0:
            t = min((epoch - 1) / self._roi_dropout_ramp_ep, 1.0)
            self.data_loader.dataset.roi_dropout = (
                self._roi_dropout_start + t * (self._roi_dropout_end - self._roi_dropout_start)
            )

        for batch_idx, batch in enumerate(self.data_loader):
            # 5-element batches (LocalGlobalDataset): (local, glob, pos, mask, global_mask)
            # 2-element batches (GlobalDataset / LocalDataset): (image, mask)
            if len(batch) > 2:
                *raw_inputs, target, aux_target = batch
                aux_target = aux_target.to(self.device) if self.aux_loss_weight > 0 else None
            else:
                *raw_inputs, target = batch
                aux_target = None
            inputs = [x.to(self.device) for x in raw_inputs]
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(*inputs)
            if isinstance(output, tuple):
                main_out, aux_out = output
                main_loss = self.criterion(main_out, target)
                if aux_target is not None:
                    loss = main_loss + self.aux_loss_weight * self.criterion(aux_out, aux_target)
                else:
                    loss = main_loss
                output = main_out  # for metrics
            else:
                main_loss = None
                loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            if self.aux_loss_weight > 0 and main_loss is not None:
                self.train_metrics.update('main_loss', main_loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self._switch_cfg and not self._switch_done:
            metric = self._switch_cfg['trigger_metric']
            key = metric if metric.startswith('val_') else f'val_{metric}'
            if log.get(key, 0.0) >= self._switch_cfg['threshold']:
                import model.loss as module_loss
                self.criterion = getattr(module_loss, self._switch_cfg['loss'])
                self._switch_done = True
                self.logger.info(
                    f"[loss_switch] {key}={log[key]:.4f} >= {self._switch_cfg['threshold']} "
                    f"-> criterion switched to {self._switch_cfg['loss']}"
                )

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                if len(batch) > 2:
                    *raw_inputs, target, _ = batch  # strip global_mask, not used in validation
                else:
                    *raw_inputs, target = batch
                inputs = [x.to(self.device) for x in raw_inputs]
                target = target.to(self.device)

                output = self.model(*inputs)
                if isinstance(output, tuple):
                    output = output[0]
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

