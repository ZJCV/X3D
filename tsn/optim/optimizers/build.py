# -*- coding: utf-8 -*-

"""
@date: 2020/11/3 上午10:49
@file: build.py
@author: zj
@description: 
"""

import torch.nn as nn

from tsn.optim import registry
from tsn.optim.optimizers.sgd import build_sgd
from tsn.optim.optimizers.adam import build_adam


def build_optimizer(cfg, model):
    assert isinstance(model, nn.Module)
    return registry.OPTIMIZERS[cfg.OPTIMIZER.NAME](cfg, model)
