# -*- coding: utf-8 -*-

"""
@date: 2020/11/3 上午10:22
@file: act_helper.py
@author: zj
@description: 
"""

import torch.nn as nn


def get_act(type):
    if type == 'ReLU':
        return nn.ReLU
    else:
        raise ValueError(f'{type} does not exists')
