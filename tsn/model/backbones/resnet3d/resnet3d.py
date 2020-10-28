# -*- coding: utf-8 -*-

"""
@date: 2020/9/28 下午4:49
@file: resnet3d.py
@author: zj
@description: 
"""

import torch.nn as nn
from torch.nn.modules.module import T

from .utility import convTxHxW, _triple, _quadruple
from .basic_block_3d import BasicBlock3d
from .bottleneck_3d import Bottleneck3d


class ResNet3d(nn.Module):
    """ResNet 3d backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Channel num of input features. Default: 3.
        spatial_strides (Sequence[int]):
            Spatial strides of residual blocks of each stage.
            Default: ``(1, 2, 2, 2)``.
        temporal_strides (Sequence[int]):
            Temporal strides of residual blocks of each stage.
            Default: ``(1, 1, 1, 1)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        conv1_kernel (Sequence[int]): Kernel size of the first conv layer.
            Default: ``(1, 7, 7)``.
        conv1_stride_t (int): Temporal stride of the first conv layer.
            Default: 2.
        pool1_stride_t (int): Temporal stride of the first pooling layer.
            Default: 2.
        with_pool2 (bool): Whether to use pool2. Default: True.
        inflate (Sequence[int]): Inflate Dims of each block.
            Default: (0, 0, 0, 0).
        inflate_style (str): ``3x1x1`` or ``1x1x1``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Default: '3x1x1'.
        non_local (Sequence[int]): Determine whether to apply non-local module
            in the corresponding block of each stages. Default: (0, 0, 0, 0).
        norm_layer (nn.Module): norm layers.
            Default: None.
        act_layer (nn.Module): activation layer.
            Default: None.
        zero_init_residual (bool):
            Whether to use zero initialization for residual block,
            Default: True.
        state_dict_2d (bool): pretrained 2D model.
            Default: None.
        partial_bn (bool): freezing all bn except the first
            Default: False
        kwargs (dict, optional): Key arguments for "make_res_layer".
    """

    arch_settings = {
        "resnet18": (BasicBlock3d, (2, 2, 2, 2)),
        "resnet34": (BasicBlock3d, (3, 4, 6, 3)),
        "resnet50": (Bottleneck3d, (3, 4, 6, 3)),
        "resnet101": (Bottleneck3d, (3, 4, 23, 3)),
        "resnet152": (Bottleneck3d, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 base_channel=64,
                 conv1_kernel=(1, 7, 7),
                 conv1_stride_t=2,
                 pool1_kernel_t=3,
                 pool1_stride_t=2,
                 with_pool2=True,
                 inflates=(0, 0, 0, 0),
                 inflate_style='3x1x1',
                 non_local=(0, 0, 0, 0),
                 norm_layer=None,
                 act_layer=None,
                 zero_init_residual=True,
                 state_dict_2d=None,
                 partial_bn=False):
        super().__init__()
        assert len(spatial_strides) == len(temporal_strides) == len(dilations) == len(inflates) == 4
        assert len(conv1_kernel) == 3

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if act_layer is None:
            act_layer = nn.ReLU

        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.with_pool2 = with_pool2
        self.base_channels = base_channel
        self._make_stem_layer(in_channels, self.base_channels, conv1_kernel, conv1_stride_t, pool1_kernel_t,
                              pool1_stride_t)

        block, stage_blocks = self.arch_settings[depth]
        self.block = block
        self.stage_blocks = stage_blocks

        self.res_layers = list()
        self.inplanes = self.base_channels
        res_planes = [base_channel, base_channel * 2, base_channel * 2 * 2, base_channel * 2 * 2 * 2]
        for i in range(len(stage_blocks)):
            res_layer = self.make_res_layer(block,
                                            res_planes[i],
                                            stage_blocks[i],
                                            spatial_stride=spatial_strides[i],
                                            temporal_stride=temporal_strides[i],
                                            dilation=dilations[i],
                                            inflate=inflates[i],
                                            inflate_style=inflate_style,
                                            non_local=non_local[i],
                                            norm_layer=self.norm_layer,
                                            act_layer=self.act_layer)

            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.zero_init_residual = zero_init_residual
        self._init_weights(state_dict_2d)

        self.partial_bn = partial_bn

    def _init_weights(self, state_dict_2d):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                if (
                        hasattr(m, "transform_final_bn")
                        and m.transform_final_bn
                ):
                    batchnorm_weight = 0.0
                else:
                    batchnorm_weight = 1.0

                nn.init.constant_(m.weight, batchnorm_weight)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3d):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock3d):
                    nn.init.constant_(m.bn2.weight, 0)

        if state_dict_2d:

            def _inflate_conv_params(conv3d, state_dict_2d, module_name_2d,
                                     inflated_param_names):
                """Inflate a conv module from 2d to 3d.

                Args:
                    conv3d (nn.Module): The destination conv3d module.
                    state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
                    module_name_2d (str): The name of corresponding conv module in the
                        2d model.
                    inflated_param_names (list[str]): List of parameters that have been
                        inflated.
                """
                weight_2d_name = module_name_2d + '.weight'

                conv2d_weight = state_dict_2d[weight_2d_name]
                kernel_t = conv3d.weight.data.shape[2]

                new_weight = conv2d_weight.data.unsqueeze(2).expand_as(
                    conv3d.weight) / kernel_t
                conv3d.weight.data.copy_(new_weight)
                inflated_param_names.append(weight_2d_name)

                if getattr(conv3d, 'bias') is not None:
                    bias_2d_name = module_name_2d + '.bias'
                    conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
                    inflated_param_names.append(bias_2d_name)

            def _inflate_bn_params(bn3d, state_dict_2d, module_name_2d,
                                   inflated_param_names):
                """Inflate a norm module from 2d to 3d.

                Args:
                    bn3d (nn.Module): The destination bn3d module.
                    state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
                    module_name_2d (str): The name of corresponding bn module in the
                        2d model.
                    inflated_param_names (list[str]): List of parameters that have been
                        inflated.
                """
                for param_name, param in bn3d.named_parameters():
                    param_2d_name = f'{module_name_2d}.{param_name}'
                    param_2d = state_dict_2d[param_2d_name]
                    param.data.copy_(param_2d)
                    inflated_param_names.append(param_2d_name)

                for param_name, param in bn3d.named_buffers():
                    param_2d_name = f'{module_name_2d}.{param_name}'
                    # some buffers like num_batches_tracked may not exist in old
                    # checkpoints
                    if param_2d_name in state_dict_2d:
                        param_2d = state_dict_2d[param_2d_name]
                        param.data.copy_(param_2d)
                        inflated_param_names.append(param_2d_name)

            inflated_param_names = []
            for name, module in self.named_modules():
                if 'non_local' in name:
                    continue
                if isinstance(module, nn.Conv3d):
                    _inflate_conv_params(module, state_dict_2d, name, inflated_param_names)
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    _inflate_bn_params(module, state_dict_2d, name, inflated_param_names)

            # check if any parameters in the 2d checkpoint are not loaded
            remaining_names = set(
                state_dict_2d.keys()) - set(inflated_param_names)
            if remaining_names:
                print(f'These parameters in the 2d checkpoint are not loaded: {sorted(remaining_names)}')

    def make_res_layer(self,
                       block,
                       planes,
                       blocks,
                       spatial_stride=1,
                       temporal_stride=1,
                       dilation=1,
                       inflate=1,
                       inflate_style='3x1x1',
                       non_local=0,
                       norm_layer=None,
                       act_layer=None):
        """Build residual layer for ResNet3D.

        Args:
            block (nn.Module): Residual module to be built.
            planes (int): Number of channels for the output feature
                in each block.
            blocks (int): Number of residual blocks.
            spatial_stride (int | Sequence[int]): Spatial strides in
                residual and conv layers. Default: 1.
            temporal_stride (int | Sequence[int]): Temporal strides in
                residual and conv layers. Default: 1.
            dilation (int): Spacing between kernel elements. Default: 1.
            inflate (int | Sequence[int]): Determine whether to inflate
                for each block. Default: 1.
            inflate_style (str): ``3x1x1`` or ``1x1x1``. which determines
                the kernel sizes and padding strides for conv1 and conv2
                in each block. Default: '3x1x1'.
            non_local (int | Sequence[int]): Determine whether to apply
                non-local module in the corresponding block of each stages.
                Default: 0.
            norm_layer (nn.Module): norm layers.
                Default: None.
            act_layer (nn.Module): activation layer.
                Default: None.
        Returns:
            nn.Module: A residual layer for the given config.
        """
        inflate = inflate if not isinstance(inflate, int) else (inflate,) * blocks
        non_local = non_local if not isinstance(non_local, int) else (non_local,) * blocks
        assert len(inflate) == blocks and len(non_local) == blocks

        downsample = None
        if spatial_stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                convTxHxW(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(temporal_stride, spatial_stride, spatial_stride),
                    padding=0,
                    bias=False,
                ),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                downsample=downsample,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                non_local=(non_local[0] == 1),
                norm_layer=norm_layer,
                act_layer=act_layer,
            ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    spatial_stride=1,
                    temporal_stride=1,
                    dilation=dilation,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    non_local=(non_local[i] == 1),
                    norm_layer=norm_layer,
                    act_layer=act_layer
                ))

        return nn.Sequential(*layers)

    def _make_stem_layer(self, inplanes, planes, conv1_kernel, conv1_stride_t, pool1_kernel_t, pool1_stride_t):
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        self.conv1 = convTxHxW(
            inplanes,
            planes,
            kernel_size=conv1_kernel,
            stride=(conv1_stride_t, 2, 2),
            padding=tuple([(k - 1) // 2 for k in _triple(conv1_kernel)]),
            bias=False
        )

        self.bn1 = self.norm_layer(planes)
        self.relu = self.act_layer(inplace=True)

        self.maxpool = nn.MaxPool3d(
            kernel_size=(pool1_kernel_t, 3, 3),
            stride=(pool1_stride_t, 2, 2),
            padding=(pool1_kernel_t // 2, 1, 1))

        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def freezing_bn(self):
        count = 0
        for m in self.modules():
            if isinstance(m, type(self.norm_layer)):
                count += 1
                if count == 1:
                    continue

                m.eval()
                # shutdown update in frozen mode
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def train(self: T, mode: bool = True) -> T:
        super(ResNet3d, self).train(mode=mode)

        if mode and self.partial_bn:
            self.freezing_bn()

        return self

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        assert len(x.shape) == 5

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i == 0 and self.with_pool2:
                x = self.pool2(x)

        return x
