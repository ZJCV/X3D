# -*- coding: utf-8 -*-

"""
@date: 2020/9/29 下午3:31
@file: i3d_head.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from tsn.model import registry


@registry.HEAD.register('I3DHead')
class I3DHead(nn.Module):

    def __init__(self, cfg):
        super(I3DHead, self).__init__()

        in_channels = cfg.MODEL.HEAD.FEATURE_DIMS
        num_classes = cfg.MODEL.HEAD.NUM_CLASSES
        dropout_rate = cfg.MODEL.HEAD.DROPOUT

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(in_channels, num_classes)
        if dropout_rate == 0.0:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(p=dropout_rate)

        self._init_weights()

    def _init_weights(self, mean=0, std=0.01, bias=0):
        """Initiate the parameters from scratch."""
        nn.init.normal_(self.fc.weight, mean, std)
        if hasattr(self.fc, 'bias') and self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, bias)

    def forward(self, x):
        x = self.avgpool(x)
        if self.dropout:
            x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
