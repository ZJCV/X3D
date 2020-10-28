# -*- coding: utf-8 -*-

"""
@date: 2020/9/25 下午1:56
@file: utility.py
@author: zj
@description: 
"""

from torch._six import container_abcs
from itertools import repeat
import torch.nn as nn


def convTx3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
    """Tx3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, bias=bias)


def convTx1x1(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False):
    """Tx1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


def convTxHxW(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
    """TxHxW convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, bias=bias)


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

_triple = _ntuple(3)
_quadruple = _ntuple(4)