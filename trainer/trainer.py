import math

import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
import torch.nn.functional as F

from model.loss import vicreg_features, dense_byol
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
                self.writer.add_image('input', make_grid(inputs[0].cpu(), nrow=8, normalize=True))

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
                self.writer.add_image('input', make_grid(inputs[0].cpu(), nrow=8, normalize=True))

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

        for batch_idx, (aug_global, pos, clean_global, local_target) in enumerate(self.data_loader):
            aug_global   = aug_global.to(self.device)
            pos          = pos.to(self.device)
            clean_global = clean_global.to(self.device)
            local_target = local_target.to(self.device)

            self.optimizer.zero_grad()
            global_recon, local_recon, feat = self.model(aug_global, pos, local_target)

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

                n = min(8, aug_global.size(0))
                vis_aug   = torch.stack([self._denorm_for_vis(aug_global[i])   for i in range(n)])
                vis_clean = torch.stack([self._denorm_for_vis(clean_global[i]) for i in range(n)])
                vis_grecon = torch.stack([self._denorm_for_vis(global_recon[i].detach()) for i in range(n)])
                self.writer.add_image('ssl/augmented_input',      make_grid(vis_aug,    nrow=n))
                self.writer.add_image('ssl/global_target',        make_grid(vis_clean,  nrow=n))
                self.writer.add_image('ssl/global_reconstruction', make_grid(vis_grecon, nrow=n))

                vis_ltarget = torch.stack([self._denorm_for_vis(local_target[i]) for i in range(n)])
                vis_lrecon  = torch.stack([self._denorm_for_vis(local_recon[i].detach()) for i in range(n)])
                self.writer.add_image('ssl/local_target',         make_grid(vis_ltarget, nrow=n))
                self.writer.add_image('ssl/local_reconstruction', make_grid(vis_lrecon,  nrow=n))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (aug_global, pos, clean_global, local_target) in enumerate(self.valid_data_loader):
                aug_global   = aug_global.to(self.device)
                pos          = pos.to(self.device)
                clean_global = clean_global.to(self.device)
                local_target = local_target.to(self.device)

                global_recon, local_recon, feat = self.model(aug_global, pos, local_target)

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


class BYOLTrainer(BaseTrainer):
    """
    Trainer for dense BYOL pretraining (BYOLDenseUNet).

    Applies EMA update to the target network after every optimizer step,
    with a cosine schedule from base_momentum to 1.0 over all epochs.
    Logs feature std to monitor for collapse.
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
        self.log_step = max(1, int(np.sqrt(data_loader.batch_size)))

        self.base_momentum = self._unwrapped().momentum
        self.total_epochs = config['trainer']['epochs']
        self.local_loss_weight = config['trainer'].get('local_loss_weight', 1.0)

        self.train_metrics = MetricTracker(
            'loss', 'byol_loss', 'local_loss', 'feat_std',
            *[m.__name__ for m in self.metric_ftns], writer=self.writer,
        )
        self.valid_metrics = MetricTracker(
            'loss', 'byol_loss', 'local_loss', 'feat_std',
            *[m.__name__ for m in self.metric_ftns], writer=self.writer,
        )

    def _unwrapped(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _current_momentum(self, epoch):
        t = (epoch - 1) / max(1, self.total_epochs)
        return 1 - (1 - self.base_momentum) * (math.cos(math.pi * t) + 1) / 2

    def _feat_std(self, feat):
        f = feat.detach().permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        return f.std(dim=0).mean().item()

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        tau = self._current_momentum(epoch)

        for batch_idx, (v1, v2, local_target, pos) in enumerate(self.data_loader):
            v1 = v1.to(self.device)
            v2 = v2.to(self.device)
            local_target = local_target.to(self.device)
            pos = pos.to(self.device)

            self.optimizer.zero_grad()
            z1, z2, t1, t2, local_recon, feat1 = self.model(v1, v2, pos)
            byol_loss = dense_byol(z1, z2, t1, t2)
            local_loss = F.mse_loss(local_recon, local_target)
            loss = byol_loss + self.local_loss_weight * local_loss
            loss.backward()
            self.optimizer.step()
            self._unwrapped().ema_update(tau)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('byol_loss', byol_loss.item())
            self.train_metrics.update('local_loss', local_loss.item())
            self.train_metrics.update('feat_std', self._feat_std(feat1))
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(local_recon, local_target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} (byol: {:.4f}, local: {:.4f}, tau: {:.4f})'.format(
                    epoch, self._progress(batch_idx), loss.item(),
                    byol_loss.item(), local_loss.item(), tau))
                n = min(8, v1.size(0))
                self.writer.add_image('byol/view1', make_grid(v1[:n].cpu(), nrow=n, normalize=True))
                self.writer.add_image('byol/view2', make_grid(v2[:n].cpu(), nrow=n, normalize=True))
                self.writer.add_image('byol/local_target', make_grid(local_target[:n].cpu(), nrow=n, normalize=True))
                self.writer.add_image('byol/local_recon', make_grid(local_recon[:n].detach().cpu(), nrow=n, normalize=True))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (v1, v2, local_target, pos) in enumerate(self.valid_data_loader):
                v1 = v1.to(self.device)
                v2 = v2.to(self.device)
                local_target = local_target.to(self.device)
                pos = pos.to(self.device)
                z1, z2, t1, t2, local_recon, feat1 = self.model(v1, v2, pos)
                byol_loss = dense_byol(z1, z2, t1, t2)
                local_loss = F.mse_loss(local_recon, local_target)
                loss = byol_loss + self.local_loss_weight * local_loss
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                self.valid_metrics.update('byol_loss', byol_loss.item())
                self.valid_metrics.update('local_loss', local_loss.item())
                self.valid_metrics.update('feat_std', self._feat_std(feat1))
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(local_recon, local_target))
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
