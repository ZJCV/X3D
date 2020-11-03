# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:52
@file: build.py
@author: zj
@description: 
"""

import numpy as np
import torch

from tsn.data.build import build_dataloader
from tsn.engine.trainer import do_train
from tsn.model.recognizers.build import build_recognizer
from tsn.model.criterions.build import build_criterion
from tsn.optim.optimizers.build import build_optimizer
from tsn.optim.lr_schedulers.build import build_lr_scheduler
from tsn.util import logging
from tsn.util.checkpoint import CheckPointer
from tsn.util.collect_env import collect_env_info
from tsn.util.distributed import init_distributed_training, get_device, get_local_rank, synchronize
from tsn.util.misc import launch_job
from tsn.util.parser import parse_train_args, load_train_config


def train(cfg):
    # Set up environment.
    init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = logging.setup_logging(__name__)
    logger.info('init start')
    arguments = {"iteration": 0}

    device = get_device(get_local_rank())
    model = build_recognizer(cfg, device)
    criterion = build_criterion(cfg, device)
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = CheckPointer(model, optimizer=optimizer, scheduler=lr_scheduler, save_dir=cfg.OUTPUT_DIR,
                                save_to_disk=True, logger=logger)
    if cfg.TRAIN.RESUME:
        logger.info('resume start')
        extra_checkpoint_data = checkpointer.load(map_location=device)
        if isinstance(extra_checkpoint_data, dict):
            arguments['iteration'] = extra_checkpoint_data['iteration']
            if cfg.LR_SCHEDULER.IS_WARMUP:
                logger.info('warmup start')
                if lr_scheduler.finished:
                    optimizer.load_state_dict(lr_scheduler.after_scheduler.optimizer.state_dict())
                else:
                    optimizer.load_state_dict(lr_scheduler.optimizer.state_dict())
                lr_scheduler.optimizer = optimizer
                lr_scheduler.after_scheduler.optimizer = optimizer
                logger.info('warmup end')
        logger.info('resume end')

    data_loader = build_dataloader(cfg, is_train=True, start_iter=arguments['iteration'])

    logger.info('init end')
    synchronize()
    do_train(cfg, arguments,
             data_loader, model, criterion, optimizer, lr_scheduler,
             checkpointer, device)


def main():
    args = parse_train_args()
    cfg = load_train_config(args)

    logger = logging.setup_logging(__name__, output_dir=cfg.OUTPUT_DIR)
    logger.info(args)

    logger.info("Environment info:\n" + collect_env_info())
    logger.info("Loaded configuration file {}".format(args.config_file))
    if args.config_file:
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    launch_job(cfg=cfg, init_method=args.init_method, func=train)


if __name__ == '__main__':
    main()
