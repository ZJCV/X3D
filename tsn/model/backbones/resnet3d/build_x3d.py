# -*- coding: utf-8 -*-

"""
@date: 2020/11/3 下午2:03
@file: build_x3d.py
@author: zj
@description: 
"""

from tsn.model.backbones.resnet3d.resnet3d import ResNet3d
from tsn.model.backbones.resnet3d.bottleneck3d import Bottleneck3d
from tsn.model import registry

from tsn.model.layers.conv_helper import get_conv
from tsn.model.layers.pool_helper import get_pool
from tsn.model.layers.norm_helper import get_norm
from tsn.model.layers.act_helper import get_act

__all__ = ['ResNet3d', ]


def _resnet(cfg, block_layer):
    conv_layer = get_conv(cfg.MODEL.CONV_LAYER)
    pool_layer = get_pool(cfg.MODEL.POOL_LAYER)
    norm_layer = get_norm(cfg.MODEL.NORM_LAYER)
    act_layer = get_act(cfg.MODEL.ACT_LAYER)

    model = ResNet3d(
        # 输入通道数
        in_channels=cfg.MODEL.BACKBONE.IN_CHANNELS,
        # Stem通道数
        base_channel=cfg.MODEL.BACKBONE.BASE_CHANNEL,
        # 第一个卷积层kernel_size
        conv1_kernel=cfg.MODEL.BACKBONE.CONV1_KERNEL,
        # 第一个卷积层步长
        conv1_stride=cfg.MODEL.BACKBONE.CONV1_STRIDE,
        # 第一个卷积层零填充
        conv1_padding=cfg.MODEL.BACKBONE.CONV1_PADDING,
        # 是否使用第一个池化层
        with_pool1=cfg.MODEL.BACKBONE.WITH_POOL1,
        # 是否使用第二个池化层
        with_pool2=cfg.MODEL.BACKBONE.WITH_POOL2,
        # 各层块个数，以R50为例
        stage_blocks=cfg.MODEL.BACKBONE.STAGE_BLOCKS,
        # 各层Block第一个卷积层的输出通道数
        res_planes=cfg.MODEL.BACKBONE.RES_PLANES,
        # 膨胀系数，以Bottleneck为例
        expansion=cfg.MODEL.BACKBONE.EXPANSION,
        # 空间步长
        spatial_strides=cfg.MODEL.BACKBONE.SPATIAL_STRIDES,
        # 是否进行膨胀
        inflates=cfg.MODEL.BACKBONE.INFLATES,
        # 膨胀类型
        inflate_style=cfg.MODEL.BACKBONE.INFLATE_STYLE,
        # 卷积层类型
        conv_layer=conv_layer,
        # 池化层类型
        pool_layer=pool_layer,
        # 归一化层类型
        norm_layer=norm_layer,
        # 激活层类型
        act_layer=act_layer,
        # 块类型
        block_layer=block_layer,
        # 是否进行残差分支零初始化
        zero_init_residual=cfg.MODEL.BACKBONE.ZERO_INIT_RESIDUAL,
        # 是否进行partialBN
        partial_bn=cfg.MODEL.BACKBONE.PARTIAL_BN,
    )
    return model


@registry.BACKBONE.register('X3D')
def x3d(cfg):
    return _resnet(cfg, Bottleneck3d)
