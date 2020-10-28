# -*- coding: utf-8 -*-

"""
@date: 2020/9/23 下午2:35
@file: batchnorm_helper.py
@author: zj
@description: 
"""

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from tsn.util import logging


def simple_group_split(world_size, rank, num_groups):
    # world_size: number of all processes
    # rank: current process ID
    # num_groups: number of groups in total, e.g. world_size=8 and you want to use 4 GPUs in a syncBN group, so num_groups=2
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(dist.new_group(rank_list[i]))
    group_size = world_size // num_groups

    logger = logging.setup_logging(__name__)
    logger.info(
        "Rank no.{} start sync BN on the process group of {}".format(rank, rank_list[rank // group_size]))
    return groups[rank // group_size]


def convert_sync_bn(model, process_group, device):
    # convert all BN layers in the model to syncBN
    for _, (child_name, child) in enumerate(model.named_children()):
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            m = nn.SyncBatchNorm.convert_sync_batchnorm(child, process_group)
            m = m.to(device=device)
            setattr(model, child_name, m)
        else:
            convert_sync_bn(child, process_group, device)
