# -*- coding: utf-8 -*-

"""
@date: 2020/9/18 下午4:18
@file: jester.py
@author: zj
@description: 
"""

import os
import numpy as np

from .evaluator.jester import JesterEvaluator
from .base_dataset import VideoRecord, BaseDataset


class JESTER(BaseDataset):

    def __init__(self,
                 *args,
                 **kwargs):
        super(JESTER, self).__init__(*args, **kwargs)

        self.start_index = 1
        self.img_prefix = ''

        self._update_video(self.annotation_dir, is_train=self.is_train)
        self._update_class()
        self._sample_frames()
        self._update_dataset()
        self._update_evaluator()

    def _update_video(self, annotation_dir, is_train=True):
        if self.type == 'Video':
            raise ValueError('Jester supports only RawFrame')

        label_path = os.path.join(annotation_dir, 'jester-v1-labels.csv')
        classes = list(np.loadtxt(label_path, dtype=np.str, delimiter=','))

        if is_train:
            anno_path = os.path.join(annotation_dir, 'jester-v1-train.csv')
        else:
            anno_path = os.path.join(annotation_dir, 'jester-v1-validation.csv')

        video_list = list()
        anno_array = np.loadtxt(anno_path, dtype=np.str, delimiter=';')
        for anno in anno_array:
            path = anno[0]
            label_name = anno[1]
            label = classes.index(label_name)

            data_path = os.path.join(self.data_dir, path)
            num_frames = len(os.listdir(data_path))

            video_list.append(VideoRecord([path, num_frames, label]))

        self.classes = classes
        self.video_list = video_list

    def _update_class(self):
        super()._update_class()

    def _update_evaluator(self):
        self.evaluator = JesterEvaluator(self.classes, topk=(1, 5))