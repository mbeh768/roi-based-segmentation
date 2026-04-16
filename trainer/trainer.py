import math
from pathlib import Path

import numpy as np
import cv2
import torch
from base import BaseTrainer
import torch.nn.functional as F

from model.loss import vicreg_features, weighted_mask_mse
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

        train_keys = ['loss', 'main_loss'] if self.aux_loss_weight > 0 else ['loss']
        self.train_metrics = MetricTracker(*train_keys, *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
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


class SSLTrainer(BaseTrainer):
    """
    Trainer for self-supervised _GlobalUNet pretraining.

    Tracks global_loss and local_loss explicitly and logs input vs reconstruction
    images to tensorboard.
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            self.len_epoch = len(self.data_loader)
        else:
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.local_loss_weight = config['trainer'].get('local_loss_weight', 1.0)
        self.feat_reg_weight = config['trainer'].get('feat_reg_weight', 0.1)
        self.save_grid_period = config['trainer'].get('save_grid_period', 0)
        self.save_grid_n = config['trainer'].get('save_grid_n', 12)
        self.global_mask_loss_weight = config['trainer'].get('global_mask_loss_weight', 1.0)

        self.train_metrics = MetricTracker(
            'loss', 'global_loss', 'local_loss', 'feat_reg',
            *[m.__name__ for m in self.metric_ftns],
            writer=self.writer,
        )
        self.valid_metrics = MetricTracker(
            'loss', 'global_loss', 'local_loss',
            *[m.__name__ for m in self.metric_ftns],
            writer=self.writer,
        )

    def _denorm_for_vis(self, img):
        """Clamp a CHW tensor to [0, 1] for tensorboard display.
        Works whether the data is raw [0,1] or ImageNet-normalised."""
        lo, hi = img.min(), img.max()
        if hi - lo < 1e-6:
            return torch.zeros_like(img)
        return (img - lo) / (hi - lo)

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (aug_global, pos, clean_global, local_target, erase_mask) in enumerate(self.data_loader):
            aug_global   = aug_global.to(self.device)
            pos          = pos.to(self.device)
            clean_global = clean_global.to(self.device)
            local_target = local_target.to(self.device)

            self.optimizer.zero_grad()
            global_recon, local_recon, feat = self.model(aug_global, pos, local_target)

            if self.global_mask_loss_weight != 1.0:
                global_loss = weighted_mask_mse(global_recon, clean_global, erase_mask.to(self.device), self.global_mask_loss_weight)
            else:
                global_loss = self.criterion(global_recon, clean_global)
            local_loss  = self.criterion(local_recon, local_target)
            feat_reg    = vicreg_features(feat)
            loss = (global_loss
                    + self.local_loss_weight * local_loss
                    + self.feat_reg_weight * feat_reg)

            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('global_loss', global_loss.item())
            self.train_metrics.update('local_loss', local_loss.item())
            self.train_metrics.update('feat_reg', feat_reg.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(global_recon, clean_global))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} (global: {:.6f}, local: {:.6f})'.format(
                    epoch, self._progress(batch_idx),
                    loss.item(), global_loss.item(), local_loss.item()))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.save_grid_period > 0 and epoch % self.save_grid_period == 0:
            self._save_grid(epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    @staticmethod
    def _to_img(t, tile_h=256, interp=cv2.INTER_LINEAR):
        t = t.detach().cpu().float()
        lo, hi = t.min(), t.max()
        t = (t - lo) / (hi - lo + 1e-8)
        arr = (t.numpy() * 255).astype(np.uint8)
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = np.repeat(arr, 3, axis=0)
            arr = arr.transpose(1, 2, 0)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return cv2.resize(arr, (tile_h, tile_h), interpolation=interp)

    @staticmethod
    def _draw_pos(img, pos):
        H, W = img.shape[:2]
        y0, x0, y1, x1 = pos.tolist()
        return cv2.rectangle(img.copy(),
                             (int(x0 * W), int(y0 * H)),
                             (int(x1 * W), int(y1 * H)),
                             (0, 255, 0), 2)

    @staticmethod
    def _label(img, text):
        cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
        return img

    def _save_grid(self, epoch):
        loader = self.valid_data_loader if self.valid_data_loader is not None else self.data_loader
        if loader is None:
            return
        out_dir = Path(self.checkpoint_dir) / 'ssl_grid'
        out_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        rows = []
        collected = 0
        with torch.no_grad():
            for aug_global, pos, clean_global, local_target, _erase_mask in loader:
                aug_global   = aug_global.to(self.device)
                pos          = pos.to(self.device)
                clean_global = clean_global.to(self.device)
                local_target = local_target.to(self.device)
                global_recon, local_recon, _ = self.model(aug_global, pos, local_target)
                B = aug_global.shape[0]
                for i in range(B):
                    if collected >= self.save_grid_n:
                        break
                    panels = [
                        self._label(self._to_img(aug_global[i]), 'aug_input'),
                        self._label(self._draw_pos(self._to_img(clean_global[i]), pos[i]), 'global_target'),
                        self._label(self._to_img(global_recon[i]), 'global_recon'),
                        self._label(self._to_img(local_target[i], interp=cv2.INTER_NEAREST), 'local_target'),
                        self._label(self._to_img(local_recon[i], interp=cv2.INTER_NEAREST), 'local_recon'),
                    ]
                    rows.append(np.concatenate(panels, axis=1))
                    collected += 1
                if collected >= self.save_grid_n:
                    break

        if rows:
            grid = np.concatenate(rows, axis=0)
            cv2.imwrite(str(out_dir / f'grid_epoch{epoch}.png'), grid)
            self.logger.info(f'Saved SSL grid ({collected} samples) for epoch {epoch}')

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (aug_global, pos, clean_global, local_target, erase_mask) in enumerate(self.valid_data_loader):
                aug_global   = aug_global.to(self.device)
                pos          = pos.to(self.device)
                clean_global = clean_global.to(self.device)
                local_target = local_target.to(self.device)

                global_recon, local_recon, feat = self.model(aug_global, pos, local_target)

                if self.global_mask_loss_weight != 1.0:
                    global_loss = weighted_mask_mse(global_recon, clean_global, erase_mask.to(self.device), self.global_mask_loss_weight)
                else:
                    global_loss = self.criterion(global_recon, clean_global)
                local_loss  = self.criterion(local_recon, local_target)
                feat_reg    = vicreg_features(feat)
                loss = (global_loss
                        + self.local_loss_weight * local_loss
                        + self.feat_reg_weight * feat_reg)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                self.valid_metrics.update('global_loss', global_loss.item())
                self.valid_metrics.update('local_loss', local_loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(global_recon, clean_global))

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
