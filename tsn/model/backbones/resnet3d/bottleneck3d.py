# -*- coding: utf-8 -*-

"""
@date: 2020/9/28 下午4:35
@file: bottleneck3d.py
@author: zj
@description: 
"""

import torch.nn as nn
from tsn.model.layers.conv_helper import convTx1x1


class Bottleneck3d(nn.Module):
    """
    Bottleneck 3d block for ResNet3D.
    """

    def __init__(self,
                 # 输入通道
                 inplanes,
                 # 输出通道
                 planes,
                 # 空间步长
                 spatial_stride=1,
                 # 是否膨胀
                 inflate=True,
                 # 膨胀类型
                 inflate_style='3x1x1',
                 # 膨胀系数
                 expansion=4,
                 # 卷积层类型
                 conv_layer=None,
                 # 归一化层类型
                 norm_layer=None,
                 # 激活层类型
                 act_layer=None):
        super().__init__()
        assert inflate_style in ['3x1x1', '3x3x3']
        if conv_layer is None:
            conv_layer = nn.Conv3d
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if act_layer is None:
            act_layer = nn.ReLU

        # 输入通道
        self.inplanes = inplanes
        # 输出通道
        self.planes = planes
        # 空间步长
        self.spatial_stride = spatial_stride
        # 是否膨胀
        self.inflate = inflate
        # 膨胀类型
        self.inflate_style = inflate_style
        # 膨胀系数
        self.expansion = expansion
        # 卷积层类型
        self.conv_layer = conv_layer
        # 归一化层类型
        self.norm_layer = norm_layer
        # 激活层类型
        self.act_layer = act_layer

        if self.inflate:
            if inflate_style == '3x1x1':
                conv1_kernel_size = (3, 1, 1)
                conv1_padding = (1, 0, 0)
                conv2_kernel_size = (1, 3, 3)
                conv2_padding = (0, 1, 1)
            else:
                conv1_kernel_size = (1, 1, 1)
                conv1_padding = (0, 0, 0)
                conv2_kernel_size = (3, 3, 3)
                conv2_padding = (1, 1, 1)
        else:
            conv1_kernel_size = (1, 1, 1)
            conv1_padding = (0, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, 1, 1)

        # Tx1x1
        self.conv1 = convTx1x1(inplanes,
                               planes,
                               kernel_size=conv1_kernel_size,
                               padding=conv1_padding,
                               bias=False)
        self.bn1 = norm_layer(planes)

        # Tx3x3
        self.conv2 = conv_layer(planes,
                                planes,
                                kernel_size=conv2_kernel_size,
                                # 是否进行空间下采样
                                stride=(1, self.spatial_stride, self.spatial_stride),
                                padding=conv2_padding,
                                bias=False)
        self.bn2 = norm_layer(planes)

        # Tx1x1
        out_planes = int(planes * self.expansion)
        self.conv3 = convTx1x1(planes,
                               out_planes,
                               kernel_size=(1, 1, 1),
                               padding=(0, 0, 0),
                               bias=False)
        self.bn3 = norm_layer(out_planes)

        self.act = self.act_layer(inplace=True)
        downsample = None
        if self.spatial_stride != 1 or self.inplanes != out_planes:
            # 下采样
            # 空间维度或者通道维度
            downsample = nn.Sequential(
                conv_layer(inplanes,
                           out_planes,
                           kernel_size=(1, 1, 1),
                           stride=(1, self.spatial_stride, self.spatial_stride),
                           padding=(0, 0, 0),
                           bias=False),
                norm_layer(out_planes),
            )
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out
