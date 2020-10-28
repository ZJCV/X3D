# -*- coding: utf-8 -*-

"""
@date: 2020/8/29 上午11:05
@file: base_dataset.py
@author: zj
@description: 
"""

from abc import ABCMeta, abstractmethod
from torch.utils.data import Dataset

from .clipsample import SegmentedSample, DenseSample
from .rawframe_dataset import RawFrameDataset
from .video_dataset import VideoDataset


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class BaseDataset(Dataset, metaclass=ABCMeta):

    def __init__(self,
                 data_dir,
                 annotation_dir,
                 modality="RGB",
                 type='RawFrame',
                 sample_strategy='SegSample',
                 clip_len=1,
                 frame_interval=1,
                 num_clips=3,
                 num_sample_positions=10,
                 is_train=True,
                 transform=None,
                 **kwargs):
        assert isinstance(modality, str) and modality in ('RGB', 'RGBDiff')
        assert isinstance(type, str) and type in ('RawFrame', 'Video')
        assert isinstance(sample_strategy, str) and sample_strategy in ('SegSample', 'DenseSample')

        if modality == 'RGB':
            assert clip_len == 1
        elif modality == 'RGBDiff':
            assert clip_len == 5
            # Diff needs one more image to calculate diff
            clip_len += 1
        else:
            raise ValueError(f'{self.modality} does not exist')

        self.data_dir = data_dir
        self.annotation_dir = annotation_dir
        self.modality = modality
        self.type = type
        self.sample_strategy = sample_strategy
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.num_sample_positions = num_sample_positions
        self.is_train = is_train
        self.transform = transform
        self.kwargs = kwargs

        if type == 'Video':
            self.enable_multithread_decode = self.kwargs['enable_multithread_decode']
            self.decoding_backend = self.kwargs['decoding_backend']

        self.video_list = None
        self.cate_list = None
        self.img_num_list = None
        self.sampler = None
        # RawFrames下标从0开始，比如UCF101/HMDB51，也有用1开始，比如JESTER
        self.start_index = 0
        # RawFrames图像命令前缀，比如UCF101/HMDB51使用img_，JESTER没有
        self.img_prefix = 'img_'
        self.evaluator = None

    @abstractmethod
    def _update_video(self, annotation_dir, is_train=True):
        pass

    @abstractmethod
    def _update_class(self):
        pass

    @abstractmethod
    def _update_evaluator(self):
        pass

    def _sample_frames(self):
        if self.sample_strategy == 'SegSample':
            self.clip_sample = SegmentedSample(self.clip_len,
                                               self.frame_interval,
                                               self.num_clips,
                                               is_train=self.is_train,
                                               start_index=self.start_index)
        elif self.sample_strategy == 'DenseSample':
            self.clip_sample = DenseSample(self.clip_len,
                                           self.frame_interval,
                                           self.num_clips,
                                           is_train=self.is_train,
                                           start_index=self.start_index,
                                           num_sample_positions=self.num_sample_positions)
        else:
            raise ValueError(f'{self.sample_strategy} does not exist')

    def _update_dataset(self):
        if self.type == 'RawFrame':
            self.data_set = RawFrameDataset(self.clip_len,
                                            self.frame_interval,
                                            self.data_dir,
                                            self.video_list,
                                            self.clip_sample,
                                            self.modality,
                                            self.img_prefix,
                                            self.transform)
        elif self.type == 'Video':
            self.data_set = VideoDataset(self.clip_len,
                                         self.data_dir,
                                         self.video_list,
                                         self.clip_sample,
                                         self.modality,
                                         self.img_prefix,
                                         self.transform,
                                         self.enable_multithread_decode,
                                         self.decoding_backend)
        else:
            raise ValueError(f'{self.type} does not exist')

    def __getitem__(self, index: int):
        """
        从选定的视频文件夹中随机选取T帧，则返回(T, C, H, W)，其中T表示num_segs
        """
        return self.data_set.__getitem__(index)

    def __len__(self) -> int:
        return self.data_set.__len__()
