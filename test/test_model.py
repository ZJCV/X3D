# -*- coding: utf-8 -*-

"""
@date: 2020/10/19 下午5:04
@file: test_model.py
@author: zj
@description: 
"""

import torch

from tsn.config import cfg
from tsn.model.build import build_model
from tsn.util.distributed import get_device


def test_tsn():
    cfg.merge_from_file('configs/tsn_r50_ucf101_rgb_raw_seg_1x1x3.yaml')

    model = build_model(cfg, 0)
    print(model)

    device = get_device(0)
    data = torch.randn((1, 3, 13, 224, 224)).to(device=device, non_blocking=True)
    outputs = model(data)
    print(outputs.shape)


if __name__ == '__main__':
    test_tsn()
