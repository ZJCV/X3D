# -*- coding: utf-8 -*-

"""
@date: 2020/11/3 上午10:50
@file: build.py
@author: zj
@description: 
"""

from torch.optim.optimizer import Optimizer

from tsn.optim import registry
from tsn.optim.lr_schedulers.gradual_warmup import GradualWarmupScheduler
from tsn.optim.lr_schedulers.step_lr import build_step_lr
from tsn.optim.lr_schedulers.multistep_lr import build_multistep_lr
from tsn.optim.lr_schedulers.cosine_annearling_lr import build_cosine_annearling_lr


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
