# -*- coding: utf-8 -*-

"""
@date: 2020/11/3 下午2:42
@file: x3d_config.py
@author: zj
@description: 
"""


def add_custom_config(_C):
    # Inflated 3D ConvNet (I3D).

    # 第5个卷积层kernel_size
    _C.MODEL.HEAD.CONV5_KERNEL = (1, 1, 1)
    # 第5个卷积层通道数
    _C.MODEL.HEAD.CONV5_CHANNELS = 192
    # 第5个池化层kernel_size
    _C.MODEL.HEAD.POOL5_KERNEL = (1, 4, 4)