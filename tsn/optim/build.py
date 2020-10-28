# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:55
@file: trainer.py
@author: zj
@description: 
"""

import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer

from . import registry
from .lr_schedulers.gradual_warmup import GradualWarmupScheduler
from .lr_schedulers.step_lr import build_step_lr
from .lr_schedulers.multistep_lr import build_multistep_lr
from .lr_schedulers.cosine_annearling_lr import build_cosine_annearling_lr
from .optimizers.sgd import build_sgd
from .optimizers.adam import build_adam


def build_optimizer(cfg, model):
    assert isinstance(model, nn.Module)
    return registry.OPTIMIZERS[cfg.OPTIMIZER.NAME](cfg, model)


def build_lr_scheduler(cfg, optimizer):
    assert isinstance(optimizer, Optimizer)
    lr_scheduler = registry.LR_SCHEDULERS[cfg.LR_SCHEDULER.NAME](cfg, optimizer)

    if cfg.LR_SCHEDULER.IS_WARMUP:
        lr_scheduler = GradualWarmupScheduler(optimizer,
                                              multiplier=cfg.LR_SCHEDULER.WARMUP.MULTIPLIER,
                                              total_epoch=cfg.LR_SCHEDULER.WARMUP.ITERATION,
                                              after_scheduler=lr_scheduler)

        optimizer.zero_grad()
        optimizer.step()
        lr_scheduler.step()

    return lr_scheduler
