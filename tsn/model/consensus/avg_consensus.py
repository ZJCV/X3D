# -*- coding: utf-8 -*-

"""
@date: 2020/8/28 下午6:47
@file: avg_consensus.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from tsn.model import registry


@registry.CONSENSU.register('AvgConsensus')
class AvgConsensus(nn.Module):

    def __init__(self, cfg):
        super(AvgConsensus, self).__init__()
        pass

    def forward(self, input, dim=0):
        assert isinstance(input, torch.Tensor)

        output = input.mean(dim=dim, keepdim=False)
        return output
