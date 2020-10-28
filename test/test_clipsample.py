# -*- coding: utf-8 -*-

"""
@date: 2020/10/10 上午9:12
@file: test_clipsample.py
@author: zj
@description: 
"""

import numpy as np

from tsn.data.datasets.clipsample import SegmentedSample, DenseSample


def test_seg_sample():
    # np.random.seed(100)

    clip_len = 3
    frame_interval = 1
    num_clips = 10
    start_index = 0

    is_train = True
    clip_sample = SegmentedSample(clip_len,
                                  frame_interval,
                                  num_clips,
                                  is_train=is_train,
                                  start_index=start_index)
    num_frames = 100
    clip_offsets = clip_sample(num_frames)
    print(clip_offsets)

    clip_sample.is_train = False
    clip_offsets = clip_sample(num_frames)
    print(clip_offsets)


def test_dense_sample():
    np.random.seed(100)

    clip_len = 1
    frame_interval = 2
    num_clips = 32
    start_index = 0

    is_train = True
    clip_sample = DenseSample(clip_len,
                              frame_interval,
                              num_clips,
                              is_train=is_train,
                              start_index=start_index)
    num_frames = 100
    clip_offsets = clip_sample(num_frames)
    print(clip_offsets)
    print(len(clip_offsets))

    clip_sample.is_train = False
    clip_offsets = clip_sample(num_frames)
    print(clip_offsets)
    print(clip_offsets.reshape(10, -1))
    print(clip_offsets.reshape(10, -1).shape)


if __name__ == '__main__':
    # test_seg_sample()
    test_dense_sample()
