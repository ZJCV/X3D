# -*- coding: utf-8 -*-

"""
@date: 2020/9/29 下午3:30
@file: nl_recognizer.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from tsn.model import registry
from tsn.model.backbones.build import build_backbone
from tsn.model.heads.build import build_head


@registry.RECOGNIZER.register('NLRecognizer')
class NLRecognizer(nn.Module):

    def __init__(self, cfg, map_location=None):
        super(NLRecognizer, self).__init__()

        self.backbone = build_backbone(cfg, map_location=map_location)
        self.head = build_head(cfg)

    def forward(self, imgs):
        assert len(imgs.shape) == 5
        N, C, T, H, W = imgs.shape[:5]

        features = self.backbone(imgs)
        probs = self.head(features)

        return {'probs': probs}
