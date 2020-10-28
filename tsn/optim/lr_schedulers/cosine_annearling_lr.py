# -*- coding: utf-8 -*-

"""
@date: 2020/9/9 下午9:49
@file: cosine_annearling_lr.py
@author: zj
@description: 
"""

import torch.optim as optim
from torch.optim.optimizer import Optimizer

from tsn.optim import registry


@registry.LR_SCHEDULERS.register('cosine_annearling_lr')
def build_cosine_annearling_lr(cfg, optimizer):
    assert isinstance(optimizer, Optimizer)

    max_iteration = cfg.TRAIN.MAX_ITER
    if cfg.LR_SCHEDULER.IS_WARMUP:
        max_iteration -= cfg.LR_SCHEDULER.WARMUP.ITERATION
    minimal_lr = cfg.LR_SCHEDULER.COSINE_ANNEALING_LR.MINIMAL_LR

    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iteration, eta_min=minimal_lr)
