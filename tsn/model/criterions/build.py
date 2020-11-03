# -*- coding: utf-8 -*-

"""
@date: 2020/11/3 上午10:30
@file: build.py
@author: zj
@description: 
"""

from tsn.model import registry
from .crossentropy_loss import CrossEntropyLoss


def build_criterion(cfg, device):
    return registry.CRITERION[cfg.MODEL.CRITERION.NAME](cfg).to(device=device)
