# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:30
@file: build.py
@author: zj
@description: 
"""

from torch.nn.parallel import DistributedDataParallel as DDP

from tsn.model.batchnorm_helper import simple_group_split, convert_sync_bn
import tsn.util.distributed as du
from tsn.util.checkpoint import CheckPointer
from tsn.util import logging
from . import registry
from .recognizers.tsn_recognizer import TSNRecognizer
from .recognizers.nl_recognizer import NLRecognizer
from .criterions.crossentropy_loss import CrossEntropyLoss


def build_model(cfg, device):
    world_size = du.get_world_size()

    model = registry.RECOGNIZER[cfg.MODEL.RECOGNIZER.NAME](cfg, map_location=device).to(device=device)

    logger = logging.setup_logging(__name__)
    if cfg.MODEL.SYNC_BN and world_size > 1:
        logger.info(
            "start sync BN on the process group of {}".format(du._LOCAL_RANK_GROUP))
        convert_sync_bn(model, du._LOCAL_PROCESS_GROUP, device)
    if cfg.MODEL.PRETRAINED != "":
        logger.info(f'load pretrained: {cfg.MODEL.PRETRAINED}')
        checkpointer = CheckPointer(model, logger=logger)
        checkpointer.load(cfg.MODEL.PRETRAINED, map_location=device)
        logger.info("finish loading model weights")

    if du.get_world_size() > 1:
        model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=True)

    return model


def build_criterion(cfg, device):
    return registry.CRITERION[cfg.MODEL.CRITERION.NAME](cfg).to(device=device)
