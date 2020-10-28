# -*- coding: utf-8 -*-

"""
@date: 2020/9/7 下午3:23
@file: build.py
@author: zj
@description: 
"""

import torch.nn as nn

from tsn.model import registry

from .avg_consensus import AvgConsensus


def build_consensus(cfg):
    return registry.CONSENSU[cfg.MODEL.CONSENSU.NAME](cfg)
