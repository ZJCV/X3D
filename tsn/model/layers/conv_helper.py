# -*- coding: utf-8 -*-

"""
@date: 2020/11/3 上午10:17
@file: conv_helper.py
@author: zj
@description: 
"""

import torch.nn as nn
from torch import Tensor
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t


def get_conv(type):
    if type == 'Conv2d':
        return nn.Conv2d
    elif type == 'Conv3d':
        return nn.Conv3d
    elif type == 'ChannelWiseConv3d':
        return ChannelWiseConv3d
    else:
        raise ValueError(f'{type} does not exists')


class ChannelWiseConv3d(nn.Module):
    """
    通道可分离卷积
    channel-wise separable convolution
    来自文章MobileNets: Efficient convolutional neural networks for mobile vision applications.
    arXiv preprint arXiv:1704.04861
    参考SlowFast实现
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_3_t,
                 stride: _size_3_t = 1,
                 padding: _size_3_t = 0,
                 bias: bool = True,
                 ) -> None:
        super(ChannelWiseConv3d, self).__init__()
        stride = stride if not isinstance(stride, int) else [stride] * 3
        padding = padding if not isinstance(padding, int) else [padding] * 3
        assert len(kernel_size) == len(stride) == len(padding) == 3

        self.conv_xy = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size[1], kernel_size[2]),
            stride=(1, stride[1], stride[2]),
            padding=(0, padding[1], padding[2]),
            bias=bias,
        )
        self.conv = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(kernel_size[0], 1, 1),
            stride=(stride[0], 1, 1),
            padding=(padding[0], 0, 0),
            bias=False,
            groups=out_channels,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_xy(x)
        x = self.conv(x)
        return x
