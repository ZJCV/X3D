# -*- coding: utf-8 -*-

"""
@date: 2020/9/29 下午3:30
@file: i3d_recognizer.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from tsn.model import registry
from tsn.model.backbones.build import build_backbone
from tsn.model.heads.build import build_head


@registry.RECOGNIZER.register('I3DRecognizer')
class I3DRecognizer(nn.Module):

    def __init__(self, cfg):
        super(I3DRecognizer, self).__init__()

        self.backbone = build_backbone(cfg)
        self.head = build_head(cfg)

    def forward(self, imgs):
        assert len(imgs.shape) == 5
        N, C, T, H, W = imgs.shape[:5]

        features = self.backbone(imgs)
        probs = self.head(features)

        return {'probs': probs}
