# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午8:00
@file: trainer.py
@author: zj
@description: 
"""

import os
import datetime
import time
import torch

from tsn.util.metric_logger import MetricLogger, update_meters
from tsn.util.distributed import is_master_proc, synchronize, get_device, get_local_rank
from tsn.util import logging
from tsn.engine.inference import do_evaluation


def do_train(cfg, arguments,
             data_loader, model, criterion, optimizer, lr_scheduler,
             checkpointer, device):
    logger = logging.setup_logging(__name__)
    meters = MetricLogger()
    summary_writer = None

    use_tensorboard = cfg.TRAIN.USE_TENSORBOARD
    log_step = cfg.TRAIN.LOG_STEP
    save_step = cfg.TRAIN.SAVE_STEP
    eval_step = cfg.TRAIN.EVAL_STEP
    max_iter = cfg.TRAIN.MAX_ITER
    start_iter = arguments['iteration']

    if is_master_proc() and use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        summary_writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))
    evaluator = data_loader.dataset.evaluator

    synchronize()
    start_training_time = time.time()
    end = time.time()
    logger.info("Start training ...")
    model.train()
    for iteration, (images, targets) in enumerate(data_loader, start_iter):
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        output_dict = model(images)
        loss_dict = criterion(output_dict, targets)
        loss = loss_dict['loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        acc_list = evaluator.evaluate_train(output_dict, targets)
        update_meters(cfg.NUM_GPUS, meters, loss_dict, acc_list)

        if iteration % len(data_loader) == 0 and hasattr(data_loader.batch_sampler, "set_epoch"):
            data_loader.batch_sampler.set_epoch(iteration)

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time)
        if iteration % log_step == 0:
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                meters.delimiter.join([
                    "iter: {iter:06d}",
                    "lr: {lr:.5f}",
                    '{meters}',
                    "eta: {eta}",
                    'mem: {mem}M',
                ]).format(
                    iter=iteration,
                    lr=optimizer.param_groups[0]['lr'],
                    meters=str(meters),
                    eta=eta_string,
                    mem=round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0),
                )
            )
        if is_master_proc():
            if summary_writer:
                global_step = iteration
                for name, meter in meters.meters.items():
                    summary_writer.add_scalar('{}/avg'.format(name), float(meter.avg),
                                              global_step=global_step)
                    summary_writer.add_scalar('{}/global_avg'.format(name), meter.global_avg,
                                              global_step=global_step)
                summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

            if save_step > 0 and iteration % save_step == 0:
                checkpointer.save("model_{:06d}".format(iteration), **arguments)
        if eval_step > 0 and iteration % eval_step == 0 and not iteration == max_iter:
            eval_results = do_evaluation(cfg, model, device, iteration=iteration)
            model.train()
            if is_master_proc() and summary_writer:
                for key, value in eval_results.items():
                    summary_writer.add_scalar(f'eval/{key}', value, global_step=iteration)

    if eval_step > 0:
        logger.info('Start final evaluating...')
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        eval_results = do_evaluation(cfg, model, device)

        if is_master_proc() and summary_writer:
            for key, value in eval_results.items():
                summary_writer.add_scalar(f'eval/{key}', value, global_step=arguments["iteration"])
            summary_writer.close()
    checkpointer.save("model_final", **arguments)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    return model
