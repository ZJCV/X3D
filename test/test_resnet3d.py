# -*- coding: utf-8 -*-

"""
@date: 2020/9/28 下午8:59
@file: test_resnet3d.py
@author: zj
@description: 
"""

import torch

from tsn.model.backbones.resnet3d.build_resnet3d import resnet3d_50
from tsn.config import cfg


def test_c2d():
    data = torch.randn(1, 3, 32, 224, 224)

    cfg.merge_from_file('configs/c2d-nl_r3d50_ucf101_rgb_224x32_dense.yaml')
    model = resnet3d_50(cfg)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 2048, 4, 7, 7)

    cfg.merge_from_file('configs/c2d_r3d50_ucf101_rgb_224x32_dense.yaml')
    model = resnet3d_50(cfg)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 2048, 4, 7, 7)


def test_i3d():
    data = torch.randn(1, 3, 32, 224, 224)

    cfg.merge_from_file('configs/i3d-3x1-nl_r3d50_ucf101_rgb_224x32_dense.yaml')
    model = resnet3d_50(cfg)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 2048, 4, 7, 7)

    cfg.merge_from_file('configs/i3d-3x1_r3d50_ucf101_rgb_224x32_dense.yaml')
    model = resnet3d_50(cfg)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 2048, 4, 7, 7)

    cfg.merge_from_file('configs/i3d-3x3-nl_r3d50_ucf101_rgb_224x32_dense.yaml')
    model = resnet3d_50(cfg)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 2048, 4, 7, 7)

    cfg.merge_from_file('configs/i3d-3x3_r3d50_ucf101_rgb_224x32_dense.yaml')
    model = resnet3d_50(cfg)
    outputs = model(data)
    print(outputs.shape)
    assert outputs.shape == (1, 2048, 4, 7, 7)


if __name__ == '__main__':
    test_c2d()
    test_i3d()
