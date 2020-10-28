# -*- coding: utf-8 -*-

"""
@date: 2020/10/9 下午8:34
@file: test_datasets.py
@author: zj
@description: 
"""

from tsn.config import cfg
from tsn.data.datasets.build import build_dataset
from tsn.data.transforms.build import build_transform


def test_ucf101_rgb():
    cfg.merge_from_file('configs/tsn_r50_ucf101_rgb_raw_seg_1x1x3.yaml')
    cfg.DATASETS.NUM_CLIPS = 8

    transform = build_transform(cfg, is_train=True)
    dataset = build_dataset(cfg, transform=transform, is_train=True)
    image, target = dataset.__getitem__(20)
    print(image.shape)
    print(target)

    assert image.shape == (3, 8, 224, 224)


def test_ucf101_rgbdiff():
    cfg.merge_from_file('configs/tsn_r50_ucf101_rgbdiff_raw_seg_1x1x3.yaml')

    transform = build_transform(cfg, is_train=True)
    dataset = build_dataset(cfg, transform=transform, is_train=True)
    image, target = dataset.__getitem__(20)
    print(image.shape)
    print(target)

    assert image.shape == (3, 15, 224, 224)


def test_hmdb51_rgb():
    cfg.merge_from_file('configs/tsn_r50_hmdb51_rgb_raw_seg_1x1x3.yaml')
    cfg.DATASETS.NUM_CLIPS = 8

    transform = build_transform(cfg, is_train=True)
    dataset = build_dataset(cfg, transform=transform, is_train=True)
    image, target = dataset.__getitem__(20)
    print(image.shape)
    print(target)

    assert image.shape == (3, 8, 224, 224)


def test_hmdb51_rgbdiff():
    cfg.merge_from_file('configs/tsn_r50_hmdb51_rgbdiff_raw_seg_1x1x3.yaml')

    transform = build_transform(cfg, is_train=True)
    dataset = build_dataset(cfg, transform=transform, is_train=True)
    image, target = dataset.__getitem__(20)
    print(image.shape)
    print(target)

    assert image.shape == (3, 15, 224, 224)


def test_jester_rgb():
    cfg.merge_from_file('configs/tsn_r50_jester_rgb_raw_seg_1x1x3.yaml')
    cfg.DATASETS.NUM_CLIPS = 8

    transform = build_transform(cfg, is_train=True)
    dataset = build_dataset(cfg, transform=transform, is_train=True)
    image, target = dataset.__getitem__(20)
    print(image.shape)
    print(target)

    assert image.shape == (3, 8, 224, 224)


def test_jester_rgbdiff():
    cfg.merge_from_file('configs/tsn_r50_jester_rgbdiff_raw_seg_1x1x3.yaml')

    transform = build_transform(cfg, is_train=True)
    dataset = build_dataset(cfg, transform=transform, is_train=True)
    image, target = dataset.__getitem__(20)
    print(image.shape)
    print(target)

    assert image.shape == (3, 15, 224, 224)


if __name__ == '__main__':
    test_ucf101_rgb()
    test_ucf101_rgbdiff()
    test_hmdb51_rgb()
    test_hmdb51_rgbdiff()
    test_jester_rgb()
    test_jester_rgbdiff()
