# -*- coding: utf-8 -*-

"""
@date: 2020/8/29 上午10:57
@file: ucf101.py
@author: zj
@description: 
"""

import os

from .evaluator.ucf101 import UCF101Evaluator
from .base_dataset import VideoRecord, BaseDataset
from .video import video_container as container
from .video import decoder
from tsn.util import logging

classes = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling',
           'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball',
           'BasketballDunk', 'BenchPress', 'Biking', 'Billiards',
           'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 'Bowling',
           'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke',
           'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', 'CricketBowling',
           'CricketShot', 'CuttingInKitchen', 'Diving', 'Drumming', 'Fencing',
           'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch',
           'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow',
           'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump',
           'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing',
           'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope',
           'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade',
           'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars',
           'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol',
           'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar',
           'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse',
           'PullUps', 'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor',
           'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput',
           'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling',
           'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing',
           'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus',
           'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking',
           'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo']


class UCF101(BaseDataset):

    def __init__(self,
                 *args,
                 split=1,
                 **kwargs):
        assert isinstance(split, int) and split in (1, 2, 3)
        super(UCF101, self).__init__(*args, **kwargs)

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
            annotation_path = os.path.join(annotation_dir, f'ucf101_train_split_{self.split}_{dataset_type}.txt')
        else:
            annotation_path = os.path.join(annotation_dir, f'ucf101_val_split_{self.split}_{dataset_type}.txt')

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
        self.evaluator = UCF101Evaluator(self.classes, topk=(1, 5))
