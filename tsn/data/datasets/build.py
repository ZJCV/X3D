# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午9:21
@file: trainer.py
@author: zj
@description: 
"""

from .hmdb51 import HMDB51
from .ucf101 import UCF101
from .jester import JESTER


def build_dataset(cfg, transform=None, is_train=True):
    modality = cfg.DATASETS.MODALITY
    sample_strategy = cfg.DATASETS.SAMPLE_STRATEGY
    clip_len = cfg.DATASETS.CLIP_LEN
    frame_interval = cfg.DATASETS.FRAME_INTERVAL
    num_clips = cfg.DATASETS.NUM_CLIPS
    num_sample_positions = cfg.DATASETS.NUM_SAMPLE_POSITIONS

    dataset_name = cfg.DATASETS.TRAIN.NAME if is_train else cfg.DATASETS.TEST.NAME
    data_dir = cfg.DATASETS.TRAIN.DATA_DIR if is_train else cfg.DATASETS.TEST.DATA_DIR
    annotation_dir = cfg.DATASETS.TRAIN.ANNOTATION_DIR if is_train else cfg.DATASETS.TEST.ANNOTATION_DIR

    if dataset_name in ('HMDB51', 'UCF101'):
        DataSet = HMDB51 if dataset_name == 'HMDB51' else UCF101
        split = cfg.DATASETS.TRAIN.SPLIT if is_train else cfg.DATASETS.TEST.SPLIT
        dataset = DataSet(data_dir,
                          annotation_dir,
                          split=split,
                          is_train=is_train,
                          transform=transform,
                          modality=modality,
                          sample_strategy=sample_strategy,
                          clip_len=clip_len,
                          frame_interval=frame_interval,
                          num_clips=num_clips,
                          num_sample_positions=num_sample_positions)
    elif dataset_name == 'JESTER':
        dataset = JESTER(data_dir,
                         annotation_dir,
                         is_train=is_train,
                         transform=transform,
                         modality=modality,
                         sample_strategy=sample_strategy,
                         clip_len=clip_len,
                         frame_interval=frame_interval,
                         num_clips=num_clips,
                         num_sample_positions=num_sample_positions)
    else:
        raise ValueError(f"the dataset {dataset_name} does not exist")

    return dataset
