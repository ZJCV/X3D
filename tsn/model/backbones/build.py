# -*- coding: utf-8 -*-

"""
@date: 2020/9/7 下午3:23
@file: build.py
@author: zj
@description: 
"""

from tsn.model import registry
from .resnet3d.build_resnet3d import resnet3d_50
from .resnet3d.build_x3d import x3d


def build_backbone(cfg):
    return registry.BACKBONE[cfg.MODEL.BACKBONE.NAME](cfg)
