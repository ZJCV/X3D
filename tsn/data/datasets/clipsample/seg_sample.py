# -*- coding: utf-8 -*-

"""
@date: 2020/10/9 下午10:43
@file: seg_sample.py
@author: zj
@description: 
"""

import numpy as np


class SegmentedSample():

    def __init__(self,
                 clip_len,
                 frame_interval,
                 num_clips,
                 is_train=True,
                 start_index=0):
        """
        离散采样
        :param clip_len: 对于RGB模态，clip_len=1；对于RGBDiff模态，clip_len=6
        :param frame_interval: 单次clip中帧间隔
        :param num_clips: 将视频帧均分成num_clips，在每个clip中随机采样clip_len帧
        :param is_train:
        :param start_index:　数据集下标从0或者1开始
        """

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.is_train = is_train
        self.start_index = start_index

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        one_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - one_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, one_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - one_clip_len + 1, size=self.num_clips))
        elif avg_interval == 0:
            ratio = (num_frames - one_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        one_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - one_clip_len + 1) / float(self.num_clips)
        if num_frames > one_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
        return clip_offsets

    def __call__(self, num_frames):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.is_train:
            clip_offsets = self._get_train_clips(num_frames)
        else:
            clip_offsets = self._get_test_clips(num_frames)

        return np.round(clip_offsets).astype(np.int) + self.start_index
