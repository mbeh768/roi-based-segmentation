import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(config):
    logger = config.get_logger('train')

    # Seed the global RNGs from the data loader's split_seed so that --seed varies
    # the train/val split AND weight init / sampling together — each run is then a
    # fully independent draw, which is what makes the across-seed spread meaningful.
    seed = config['data_loader']['args'].get('split_seed', SEED)
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.info('seed=%d (split + init)', seed)

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    # loss may be a plain string "bce_dice" or a dict {"type": "bce_dice", "args": {"a": 0.5}}
    import functools
    _loss_cfg = config['loss']
    if isinstance(_loss_cfg, dict):
        criterion = functools.partial(getattr(module_loss, _loss_cfg['type']),
                                      **_loss_cfg.get('args', {}))
    else:
        criterion = getattr(module_loss, _loss_cfg)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    # optimizer config may carry an optional "encoder_lr_scale" for discriminative
    # fine-tuning of unfrozen local-encoder stages. init_obj ignores this sibling
    # key (it only reads "type"/"args"). Accepts either:
    #   - a scalar  -> every trainable backbone param trains at base_lr * scale
    #   - a dict    -> per-stage scales keyed by backbone stage index, e.g.
    #                  {"4": 0.5, "3": 0.25}; stages absent from the dict (but
    #                  still trainable) default to scale 1.0 (base_lr).
    # The decoder/head always train at base_lr. Stage->module mapping comes from
    # the model's _STAGE_MODULES, so adding more unfrozen stages with their own
    # lrs is purely a config change.
    enc_lr_scale = config['optimizer'].get('encoder_lr_scale', None)
    if enc_lr_scale is not None:
        base_lr = config['optimizer']['args']['lr']
        core = model.module if isinstance(model, torch.nn.DataParallel) else model
        stage_modules = getattr(core, '_STAGE_MODULES', None)
        if stage_modules is None or not hasattr(core, 'backbone'):
            raise ValueError(
                "encoder_lr_scale is set but model exposes no backbone/_STAGE_MODULES "
                "to map encoder stages to parameters")

        # map each backbone param id -> its stage index
        stage_of = {}
        for stage, mod_names in stage_modules.items():
            for mod_name in mod_names:
                mod = getattr(core.backbone, mod_name, None)
                if mod is None:
                    continue
                for p in mod.parameters():
                    stage_of[id(p)] = stage

        scale_is_dict = isinstance(enc_lr_scale, dict)
        scale_by_stage = ({int(k): float(v) for k, v in enc_lr_scale.items()}
                          if scale_is_dict else None)

        other_params = []
        by_scale = {}  # scale -> list of params (groups merged by identical lr)
        for p in model.parameters():
            if not p.requires_grad:
                continue
            stage = stage_of.get(id(p), None)
            if stage is None:
                other_params.append(p)              # decoder / head
            elif scale_is_dict:
                scale = scale_by_stage.get(stage, 1.0)
                by_scale.setdefault(scale, []).append(p)
            else:
                by_scale.setdefault(float(enc_lr_scale), []).append(p)

        trainable_params = [{'params': other_params}]   # base lr group
        for scale in sorted(by_scale):
            trainable_params.append({'params': by_scale[scale],
                                     'lr': base_lr * scale})
            logger.info('encoder_lr_scale: %d params at lr=%g (scale %g)',
                        len(by_scale[scale]), base_lr * scale, scale)
        logger.info('%d decoder/head params at base lr=%g',
                    len(other_params), base_lr)
    else:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--seed'], type=int, target='data_loader;args;split_seed'),
        CustomArgs(['--fold'], type=int, target='data_loader;args;fold'),
        CustomArgs(['--n_folds'], type=int, target='data_loader;args;n_folds')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
