# -*- coding: utf-8 -*-

"""
@date: 2020/8/28 下午8:52
@file: image.py
@author: zj
@description: 
"""

import cv2
import numpy as np


def rgbdiff(img1, img2, scale=True):
    assert len(img1.shape) == 3 and len(img2.shape) == 3

    diff = np.abs(img1.astype(np.float) - img2.astype(np.float))

    if scale:
        min_value = np.min(diff)
        max_value = np.max(diff)

        if max_value < 1:
            max_value = 1
        diff = (diff - min_value) / (max_value - min_value) * 255

    return diff.astype(np.uint8)
