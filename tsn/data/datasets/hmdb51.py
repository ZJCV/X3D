# -*- coding: utf-8 -*-

"""
@date: 2020/8/28 下午4:37
@file: hmdb51.py
@author: zj
@description: 
"""

import os

from .evaluator.hmdb51 import HMDB51Evaluator
from .base_dataset import VideoRecord, BaseDataset
from .video import video_container as container
from .video import decoder
from tsn.util import logging

classes = ['brush_hair', 'cartwheel', 'catch', 'chew', 'clap', 'climb',
           'climb_stairs', 'dive', 'draw_sword', 'dribble', 'drink', 'eat',
           'fall_floor', 'fencing', 'flic_flac', 'golf', 'handstand', 'hit',
           'hug', 'jump', 'kick', 'kick_ball', 'kiss', 'laugh', 'pick',
           'pour', 'pullup', 'punch', 'push', 'pushup', 'ride_bike',
           'ride_horse', 'run', 'shake_hands', 'shoot_ball', 'shoot_bow',
           'shoot_gun', 'sit', 'situp', 'smile', 'smoke', 'somersault',
           'stand', 'swing_baseball', 'sword', 'sword_exercise', 'talk',
           'throw', 'turn', 'walk', 'wave']


class HMDB51(BaseDataset):

    def __init__(self,
                 *args,
                 split=1,
                 **kwargs):
        assert isinstance(split, int) and split in (1, 2, 3)
        super(HMDB51, self).__init__(*args, **kwargs)

        self.split = split
        self.start_index = 0
        self.img_prefix = 'img_'

        self._update_video(self.annotation_dir, is_train=self.is_train)
        self._update_class()
        self._sample_frames()
        self._update_dataset()
        self._update_evaluator()

    def _update_video(self, annotation_dir, is_train=True):
        dataset_type = 'rawframes' if self.type == 'RawFrame' else 'videos'
        if is_train:
            annotation_path = os.path.join(annotation_dir, f'hmdb51_train_split_{self.split}_{dataset_type}.txt')
        else:
            annotation_path = os.path.join(annotation_dir, f'hmdb51_val_split_{self.split}_{dataset_type}.txt')

        if not os.path.isfile(annotation_path):
            raise ValueError(f'{annotation_path}不是文件路径')

        if self.type == 'RawFrame':
            self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(annotation_path)]
        elif self.type == 'Video':
            video_list = list()
            for x in open(annotation_path):
                video_path, cate = x.strip().split(' ')
                video_path = os.path.join(self.data_dir, video_path)

                # Try to decode and sample a clip from a video.
                video_container = None
                try:
                    video_container = container.get_video_container(
                        video_path,
                        self.enable_multithread_decode,
                        self.decoding_backend,
                    )
                except Exception as e:
                    logger = logging.setup_logging(__name__)
                    logger.info(
                        "Failed to load video from {} with error {}".format(
                            video_path, e
                        )
                    )

                frames_length = decoder.get_video_length(video_container)
                video_list.append(VideoRecord([video_path, frames_length, cate]))
            self.video_list = video_list
        else:
            raise ValueError(f'{self.type} does not exist')

    def _update_class(self):
        self.classes = classes

    def _update_evaluator(self):
        self.evaluator = HMDB51Evaluator(self.classes, topk=(1, 5))
