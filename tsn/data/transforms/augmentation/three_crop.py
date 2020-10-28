# -*- coding: utf-8 -*-

"""
@date: 2020/10/19 下午4:37
@file: three_crop.py
@author: zj
@description: 
"""

import numbers
import torchvision.transforms.functional as F


class ThreeCrop(object):
    """Crop the given PIL Image into left/center/right or up/center/down

    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.

    Args:
         size (sequence or int): Desired output size of the crop. If size is an ``int``
            instead of sequence like (h, w), a square crop of size (size, size) is made.
    """

    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return three_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


def three_crop(img, size):
    """Crop the given PIL Image into left/center/right or up/center/down.

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.

    Returns:
       tuple: tuple (tl, tr, bl, br, center)
                Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    image_width, image_height = img.size
    crop_height, crop_width = size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    center = F.center_crop(img, (crop_height, crop_width))
    if image_height > image_width:
        # crop up/center/down
        left = int(round((image_width - crop_width) / 2.))
        crop_top = img.crop((left, 0, left + crop_width, crop_height))

        top = int(round(image_height - crop_height))
        crop_down = img.crop((left, top, left + crop_width, image_height))
        res = (crop_top, center, crop_down)
    else:
        # crop left/center/right
        top = int(round(image_height - crop_height) / 2.)
        crop_left = img.crop((0, top, crop_width, top + crop_height))

        left = int(round(image_width - crop_width))
        crop_right = img.crop((left, top, image_width, top + crop_height))
        res = (crop_left, center, crop_right)

    return res


if __name__ == '__main__':
    model = ThreeCrop((255, 255))

    from PIL import Image
    import numpy as np

    img = Image.fromarray(np.arange(255 * 320).reshape(255, 320).astype(np.uint8))
    crop = model(img)
    print(len(crop))
