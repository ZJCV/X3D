#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


def add_custom_config(_C):
    _C.MODEL.BACKBONE.TYPE = 'C2D'
    _C.MODEL.BACKBONE.IN_CHANNELS = 3
    _C.MODEL.BACKBONE.SPATIAL_STRIDES = (1, 2, 2, 2)
    _C.MODEL.BACKBONE.TEMPORAL_STRIDES = (1, 1, 1, 1)
    _C.MODEL.BACKBONE.DILATIONS = (1, 1, 1, 1)
    _C.MODEL.BACKBONE.BASE_CHANNEL = 64
    _C.MODEL.BACKBONE.CONV1_KERNEL = (1, 7, 7)
    _C.MODEL.BACKBONE.CONV1_STRIDE_T = 2
    _C.MODEL.BACKBONE.POOL1_KERNEL_T = 3
    _C.MODEL.BACKBONE.POOL1_STRIDE_T = 2
    _C.MODEL.BACKBONE.WITH_POOL2 = True
    _C.MODEL.BACKBONE.INFLATES = (0, 0, 0, 0)
    _C.MODEL.BACKBONE.INFLATE_STYLE = '3x1x1'
    _C.MODEL.BACKBONE.NON_LOCAL = (0, 0, 0, 0)
