# -*- coding: utf-8 -*-

"""
@date: 2020/9/28 下午4:35
@file: bottleneck_3d.py
@author: zj
@description: 
"""

import torch.nn as nn
from .utility import convTx1x1, convTx3x3
from tsn.model.nonlocal_helper import Nonlocal3d


class Bottleneck3d(nn.Module):
    """Bottleneck 3d block for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        spatial_stride (int): Spatial stride in the conv3d layer. Default: 1.
        temporal_stride (int): Temporal stride in the conv3d layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        downsample (nn.Module | None): Downsample layer. Default: None.
        inflate (bool): Whether to inflate kernel. Default: True.
        inflate_style (str): ``3x1x1`` or ``1x1x1``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Default: '3x1x1'.
        non_local (bool): Determine whether to apply non-local module in this
            block. Default: False.
        norm_layer (nn.Module): norm layers.
            Default: None.
        act_layer (nn.Module): activation layer.
            Default: None.
    """
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 inflate=True,
                 inflate_style='3x1x1',
                 non_local=False,
                 norm_layer=None,
                 act_layer=None):
        super().__init__()
        assert inflate_style in ['3x1x1', '3x3x3']
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if act_layer is None:
            act_layer = nn.ReLU

        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.inflate = inflate
        self.inflate_style = inflate_style
        self.norm_layer = norm_layer
        self.act_layer = act_layer

        self.conv1_stride_s = 1
        self.conv2_stride_s = spatial_stride
        self.conv1_stride_t = 1
        self.conv2_stride_t = temporal_stride

        if self.inflate:
            if inflate_style == '3x1x1':
                conv1_kernel_size = (3, 1, 1)
                conv1_padding = (1, 0, 0)
                conv2_kernel_size = (1, 3, 3)
                conv2_padding = (0, dilation, dilation)
            else:
                conv1_kernel_size = (1, 1, 1)
                conv1_padding = (0, 0, 0)
                conv2_kernel_size = (3, 3, 3)
                conv2_padding = (1, dilation, dilation)
        else:
            conv1_kernel_size = (1, 1, 1)
            conv1_padding = (0, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, dilation, dilation)

        self.conv1 = convTx1x1(inplanes, planes, kernel_size=conv1_kernel_size,
                               stride=(self.conv1_stride_t, self.conv1_stride_s, self.conv1_stride_s),
                               padding=conv1_padding,
                               bias=False)
        self.bn1 = norm_layer(planes)

        self.conv2 = convTx3x3(planes, planes, kernel_size=conv2_kernel_size,
                               stride=(self.conv2_stride_t, self.conv2_stride_s, self.conv2_stride_s),
                               padding=conv2_padding,
                               dilation=(1, dilation, dilation),
                               bias=False)
        self.bn2 = norm_layer(planes)

        self.conv3 = convTx1x1(planes, planes * self.expansion, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = self.act_layer(inplace=True)
        self.downsample = downsample

        self.non_local = non_local
        if self.non_local:
            self.non_local_block = Nonlocal3d(planes * self.expansion, planes * self.expansion // 2)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.non_local:
            out = self.non_local_block(out)

        return out
