# -*- coding: utf-8 -*-

"""
@date: 2020/11/3 上午9:40
@file: build_resnet3d.py
@author: zj
@description: 
"""

from torchvision.models.utils import load_state_dict_from_url

from .resnet3d import ResNet3d
from .bottleneck3d import Bottleneck3d
from tsn.model import registry

from tsn.util.distributed import get_device, get_local_rank
from tsn.model.layers.conv_helper import get_conv
from tsn.model.layers.pool_helper import get_pool
from tsn.model.layers.norm_helper import get_norm
from tsn.model.layers.act_helper import get_act

__all__ = ['ResNet3d', 'resnet3d_50', 'resnet3d_101',
           'resnet3d_152', ]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def _load_pretrained(arch, map_location=None):
    state_dict_2d = load_state_dict_from_url(model_urls[arch],
                                             progress=True,
                                             map_location=map_location)
    return state_dict_2d


def _resnet(arch, cfg, block_layer):
    pretrained2d = cfg.MODEL.BACKBONE.TORCHVISION_PRETRAINED
    state_dict_2d = None
    if pretrained2d:
        device = get_device(local_rank=get_local_rank())
        state_dict_2d = _load_pretrained(arch, map_location=device)

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
        # 第一个池化层kernel_size
        pool1_kernel=cfg.MODEL.BACKBONE.POOL1_KERNEL,
        # 第一个池化层步长
        pool1_stride=cfg.MODEL.BACKBONE.POOL1_STRIDE,
        # 是否使用第二个池化层
        with_pool2=cfg.MODEL.BACKBONE.WITH_POOL2,
        # 第二个池化层kernel_size
        pool2_kernel=cfg.MODEL.BACKBONE.POOL2_KERNEL,
        # 第二个池化层步长
        pool2_stride=cfg.MODEL.BACKBONE.POOL2_STRIDE,
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
        # 是否加载预训练模型
        state_dict_2d=state_dict_2d,
        # 是否进行partialBN
        partial_bn=cfg.MODEL.BACKBONE.PARTIAL_BN)
    return model


@registry.BACKBONE.register('R3D50')
def resnet3d_50(cfg):
    return _resnet("resnet50", cfg, Bottleneck3d)


@registry.BACKBONE.register('R3D101')
def resnet3d_101(cfg):
    return _resnet("resnet101", cfg, Bottleneck3d)


@registry.BACKBONE.register('R3D152')
def resnet3d_152(cfg):
    return _resnet("resnet152", cfg, Bottleneck3d)
