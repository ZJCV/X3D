# -*- coding: utf-8 -*-

"""
@date: 2020/10/19 下午7:49
@file: to_tensor.py
@author: zj
@description: 
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic):
        """
        Args:
            pic : Image to be converted to tensor.
            1. if pic == (PIL Image or numpy.ndarray)
            2. if pic == tuple

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, tuple):
            return torch.stack([transforms.ToTensor()(crop) for crop in pic])
        else:
            return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'
