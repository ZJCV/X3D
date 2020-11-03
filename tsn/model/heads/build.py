# -*- coding: utf-8 -*-

"""
@date: 2020/9/7 下午3:23
@file: build.py
@author: zj
@description: 
"""

from tsn.model import registry

from .tsn_head import TSNHead
from .i3d_head import I3DHead
from .x3d_head import X3DHead


def build_head(cfg):
    return registry.HEAD[cfg.MODEL.HEAD.NAME](cfg)
