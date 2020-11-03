# -*- coding: utf-8 -*-

"""
@date: 2020/11/3 上午10:17
@file: conv_helper.py
@author: zj
@description: 
"""

import torch.nn as nn


def get_pool(type):
    if type == 'MaxPool2d':
        return nn.MaxPool2d
    elif type == 'MaxPool3d':
        return nn.MaxPool3d
    else:
        raise ValueError(f'{type} does not exists')
