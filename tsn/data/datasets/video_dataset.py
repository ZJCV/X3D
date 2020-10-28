# -*- coding: utf-8 -*-

"""
@date: 2020/10/14 下午7:52
@file: video_dataset.py
@author: zj
@description: 
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .video import video_container as container
from .video import decoder
from tsn.util import logging

logger = logging.get_logger(__name__)


class VideoDataset(Dataset):

    def __init__(self,
                 clip_len,
                 data_dir,
                 video_list,
                 clip_sample,
                 modality='RGB',
                 img_prefix='img_',
                 transform=None,
                 enable_multithread_decode=False,
                 decoding_backend='pyav'):
        self.clip_len = clip_len
        self.data_dir = data_dir
        self.video_list = video_list
        self.clip_sample = clip_sample
        self.modality = modality
        self.img_prefix = img_prefix
        self.transform = transform
        self.enable_multithread_decode = enable_multithread_decode
        self.decoding_backend = decoding_backend

    def __getitem__(self, index: int):
        assert index < len(self.video_list)
        record = self.video_list[index]
        target = record.label
        video_path = os.path.join(self.data_dir, record.path)

        clip_offsets = self.clip_sample(record.num_frames)

        # Try to decode and sample a clip from a video.
        video_container = None
        try:
            video_container = container.get_video_container(
                video_path,
                self.enable_multithread_decode,
                self.decoding_backend,
            )
        except Exception as e:
            logger.info(
                "Failed to load video from {} with error {}".format(
                    video_path, e
                )
            )
        video_start_pt = min(clip_offsets)
        video_end_pt = max(clip_offsets)
        frames = decoder.get_video_frames(video_container, video_start_pt, video_end_pt)

        image_list = list()
        for offset in clip_offsets:
            if 'RGB' == self.modality:
                img = frames[offset - video_start_pt]

                if self.transform:
                    img = self.transform(img)
                image_list.append(img)
            if 'RGBDiff' == self.modality:
                tmp_list = list()
                for clip in range(self.clip_len):
                    img = frames[offset + clip - video_start_pt]

                    tmp_list.append(img)
                for clip in reversed(range(1, self.clip_len)):
                    img = tmp_list[clip] - tmp_list[clip - 1]
                    if self.transform:
                        img = self.transform(img)
                    image_list.append(img)
        # [T, C, H, W] -> [C, T, H, W]
        image = torch.stack(image_list).transpose(0, 1)

        return image, target

    def __len__(self) -> int:
        return len(self.video_list)
