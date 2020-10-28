# -*- coding: utf-8 -*-

"""
@date: 2020/8/22 下午9:40
@file: trainer.py
@author: zj
@description: 
"""

import torchvision.transforms as transforms
from .augmentation import RandomResize, ThreeCrop, ToTensor, Normalize


def build_transform(cfg, is_train=True):
    MEAN = cfg.TRANSFORM.MEAN
    STD = cfg.TRANSFORM.STD

    aug_list = list()
    aug_list.append(transforms.ToPILImage())
    if is_train:
        min, max = cfg.TRANSFORM.TRAIN.SCALE_JITTER
        assert max > 0 and min > 0 and max > min
        aug_list.append(RandomResize(min, max))
        if cfg.TRANSFORM.TRAIN.RANDOM_HORIZONTAL_FLIP:
            aug_list.append(transforms.RandomHorizontalFlip())
        if cfg.TRANSFORM.TRAIN.COLOR_JITTER is not None:
            brightness, contrast, saturation, hue = cfg.TRANSFORM.TRAIN.COLOR_JITTER
            aug_list.append(
                transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue))
        if cfg.TRANSFORM.TRAIN.RANDOM_ROTATION > 0:
            random_rotation = cfg.TRANSFORM.TRAIN.RANDOM_ROTATION
            aug_list.append(transforms.RandomRotation(random_rotation))
        if cfg.TRANSFORM.TRAIN.RANDOM_CROP:
            crop_size = cfg.TRANSFORM.TRAIN.TRAIN_CROP_SIZE
            aug_list.append(transforms.RandomCrop(crop_size))
        if cfg.TRANSFORM.TRAIN.CENTER_CROP:
            crop_size = cfg.TRANSFORM.TRAIN.TRAIN_CROP_SIZE
            aug_list.append(transforms.CenterCrop(crop_size))

        aug_list.append(ToTensor())
        aug_list.append(Normalize(MEAN, STD))

        if cfg.TRANSFORM.TRAIN.RANDOM_ERASING:
            aug_list.append(transforms.RandomErasing())
    else:
        shorter_side = cfg.TRANSFORM.TEST.SHORTER_SIDE
        assert shorter_side > 0
        aug_list.append(transforms.Resize(shorter_side))
        if cfg.TRANSFORM.TEST.CENTER_CROP:
            crop_size = cfg.TRANSFORM.TEST.TEST_CROP_SIZE
            aug_list.append(transforms.CenterCrop(crop_size))
        if cfg.TRANSFORM.TEST.THREE_CROP:
            crop_size = cfg.TRANSFORM.TEST.TEST_CROP_SIZE
            aug_list.append(ThreeCrop(crop_size))

        aug_list.append(ToTensor())
        aug_list.append(Normalize(MEAN, STD))

    transform = transforms.Compose(aug_list)
    return transform