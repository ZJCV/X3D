# -*- coding: utf-8 -*-

"""
@date: 2020/10/14 下午7:32
@file: rawframe_dataset.py
@author: zj
@description: 
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class RawFrameDataset(Dataset):

    def __init__(self,
                 clip_len,
                 frame_interval,
                 data_dir,
                 video_list,
                 clip_sample,
                 modality='RGB',
                 img_prefix='img_',
                 transform=None):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.data_dir = data_dir
        self.video_list = video_list
        self.clip_sample = clip_sample
        self.modality = modality
        self.img_prefix = img_prefix
        self.transform = transform

    def __getitem__(self, index: int):
        assert index < len(self.video_list)
        record = self.video_list[index]
        target = record.label

        clip_offsets = self.clip_sample(record.num_frames)

        video_path = os.path.join(self.data_dir, record.path)
        image_list = list()
        for offset in clip_offsets:
            if 'RGB' == self.modality:
                img_path = os.path.join(video_path, '{}{:0>5d}.jpg'.format(self.img_prefix, offset))
                img = np.array(Image.open(img_path))

                if self.transform:
                    img = self.transform(img)
                if len(img.shape) == 4:
                    image_list.extend(img)
                else:
                    image_list.append(img)
            if 'RGBDiff' == self.modality:
                tmp_list = list()
                clip_idxs = offset + \
                            np.linspace(0, self.clip_len * self.frame_interval, num=self.clip_len, dtype=np.int)
                for idx in clip_idxs:
                    img_path = os.path.join(video_path, '{}{:0>5d}.jpg'.format(self.img_prefix, idx))
                    img = np.array(Image.open(img_path))

                    tmp_list.append(img)
                for clip in reversed(range(1, self.clip_len)):
                    img = tmp_list[clip] - tmp_list[clip - 1]
                    if self.transform:
                        img = self.transform(img)
                    if len(img.shape) == 4:
                        image_list.extend(img)
                    else:
                        image_list.append(img)
        # [T, C, H, W] -> [C, T, H, W]
        image = torch.stack(image_list).transpose(0, 1)

        return image, target

    def __len__(self) -> int:
        return len(self.video_list)
