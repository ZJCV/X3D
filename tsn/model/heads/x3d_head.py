# -*- coding: utf-8 -*-

"""
@date: 2020/11/3 下午3:15
@file: x3d_head.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from tsn.model import registry
from tsn.model.layers.conv_helper import convTx1x1
from tsn.model.layers.pool_helper import get_pool
from tsn.model.layers.norm_helper import get_norm
from tsn.model.layers.act_helper import get_act


@registry.HEAD.register('X3DHead')
class X3DHead(nn.Module):

    def __init__(self, cfg):
        super(X3DHead, self).__init__()

        pool_layer = get_pool(cfg.MODEL.POOL_LAYER)
        norm_layer = get_norm(cfg.MODEL.NORM_LAYER)
        act_layer = get_act(cfg.MODEL.ACT_LAYER)

        in_channels = cfg.MODEL.HEAD.FEATURE_DIMS
        num_classes = cfg.MODEL.HEAD.NUM_CLASSES
        dropout_rate = cfg.MODEL.HEAD.DROPOUT
        conv5_channels = cfg.MODEL.HEAD.CONV5_CHANNELS
        pool5_kernel = cfg.MODEL.HEAD.POOL5_KERNEL

        self.conv5 = convTx1x1(in_channels,
                                conv5_channels,
                                kernel_size=1,
                                padding=0,
                                bias=False)
        self.bn5 = norm_layer(conv5_channels)
        self.act = act_layer(inplace=True)
        self.pool5 = pool_layer(kernel_size=pool5_kernel,
                                stride=1,
                                ceil_mode=True)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(conv5_channels, num_classes)

        if dropout_rate == 0.0:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(p=dropout_rate)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                batchnorm_weight = 1.0

                nn.init.constant_(m.weight, batchnorm_weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(self.fc.weight, mean=0, std=0.01)
                if hasattr(self.fc, 'bias') and self.fc.bias is not None:
                    nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act(x)
        x = self.pool5(x)

        x = self.avgpool(x)
        if self.dropout:
            x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
