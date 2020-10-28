# -*- coding: utf-8 -*-

"""
@date: 2020/9/7 下午3:23
@file: build.py
@author: zj
@description: 
"""

from tsn.model import registry

from .tsn_head import TSNHead
from .nl_head import NLHead


def build_head(cfg):
    return registry.HEAD[cfg.MODEL.HEAD.NAME](cfg)
