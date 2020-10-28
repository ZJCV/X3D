# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:20
@file: build.py
@author: zj
@description: 
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .datasets.build import build_dataset
from .samplers import IterationBasedBatchSampler
from .transforms.build import build_transform
import tsn.util.distributed as du


def build_dataloader(cfg,
                     is_train=True,
                     start_iter=0):
    transform = build_transform(cfg, is_train=is_train)
    dataset = build_dataset(cfg, transform=transform, is_train=is_train)

    world_size = du.get_world_size()
    num_gpus = cfg.NUM_GPUS
    rank = du.get_rank()
    if is_train:
        batch_size = cfg.DATALOADER.TRAIN_BATCH_SIZE

        if num_gpus > 1:
            sampler = DistributedSampler(dataset,
                                         num_replicas=world_size,
                                         rank=rank,
                                         shuffle=True)
        else:
            sampler = torch.utils.data.RandomSampler(dataset)
    else:
        batch_size = cfg.DATALOADER.TEST_BATCH_SIZE
        if num_gpus > 1:
            sampler = DistributedSampler(dataset,
                                         num_replicas=world_size,
                                         rank=rank,
                                         shuffle=False)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)

    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler,
                                                          batch_size=batch_size,
                                                          drop_last=False)
    if is_train:
        batch_sampler = IterationBasedBatchSampler(batch_sampler,
                                                   num_iterations=cfg.TRAIN.MAX_ITER,
                                                   start_iter=start_iter)

    data_loader = DataLoader(dataset,
                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                             batch_sampler=batch_sampler,
                             pin_memory=True)

    return data_loader
