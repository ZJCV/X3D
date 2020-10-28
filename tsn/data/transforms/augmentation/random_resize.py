# -*- coding: utf-8 -*-

"""
@date: 2020/9/25 下午4:53
@file: random_resize.py
@author: zj
@description:
"""

import numpy as np
from PIL import Image
import torchvision.transforms.functional as F

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class RandomResize(object):
    """Resize the input PIL Image to the given size.

    Args:
        min (int): Desired output size. the min of smaller edge of the image
        max (int): Desired output size. the max of smaller edge of the image. Note, max > min
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, min, max, interpolation=Image.BILINEAR):
        assert isinstance(min, int) and isinstance(max, int) and max > min
        self.min = min
        self.max = max
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        size = np.random.randint(self.min, self.max)

        return F.resize(img, size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(min={0}, max={1}, interpolation={2})'.format(self.min, self.max,
                                                                                        interpolate_str)
