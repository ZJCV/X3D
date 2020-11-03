#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Inflated 3D ConvNet (I3D).

    # Stem通道数
    _C.MODEL.BACKBONE.BASE_CHANNEL = 64
    # 第一个卷积层kernel_size
    _C.MODEL.BACKBONE.CONV1_KERNEL = (1, 7, 7)
    # 第一个卷积层步长
    _C.MODEL.BACKBONE.CONV1_STRIDE = (2, 2, 2)
    # 第一个卷积层零填充
    _C.MODEL.BACKBONE.CONV1_PADDING = (0, 3, 3)
    # 是否使用第一个池化层
    _C.MODEL.BACKBONE.WITH_POOL1 = True
    # 第一个池化层kernel_size
    _C.MODEL.BACKBONE.POOL1_KERNEL = (3, 3, 3)
    # 第一个池化层步长
    _C.MODEL.BACKBONE.POOL1_STRIDE = (2, 2, 2)
    # 是否使用第二个池化层
    _C.MODEL.BACKBONE.WITH_POOL2 = True
    # 第二个池化层kernel_size
    _C.MODEL.BACKBONE.POOL2_KERNEL = (3, 1, 1)
    # 第二个池化层步长
    _C.MODEL.BACKBONE.POOL2_STRIDE = (2, 1, 1)
    # 各层块个数，以R50为例
    _C.MODEL.BACKBONE.STAGE_BLOCKS = (3, 4, 6, 3)
    # 各层Block第一个卷积层的输出通道数
    _C.MODEL.BACKBONE.RES_PLANES = [64, 128, 256, 512]
    # 膨胀系数，以Bottleneck为例
    _C.MODEL.BACKBONE.EXPANSION = 4.
    # 空间步长
    _C.MODEL.BACKBONE.SPATIAL_STRIDES = (1, 2, 2, 2)
    # 是否进行膨胀
    _C.MODEL.BACKBONE.INFLATES = (0, 0, 0, 0)
    # 膨胀类型
    _C.MODEL.BACKBONE.INFLATE_STYLE = '3x1x1'
