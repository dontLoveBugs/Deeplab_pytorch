# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/30 23:34
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import torch
import time

from metrics import AverageMeter, Result
import utils
from criteria import cross_entropy2d as criterion


def train(args, train_loader, model, optimizer, epoch, logger):
    average_meter = AverageMeter()
    model.train()  # switch to train mode
    batch_num = len(train_loader)

    output_directory = utils.get_output_directory(args, check=True)

    end = time.time()

    for i, samples in enumerate(train_loader):

        input = samples['image']
        target = samples['label']

        # print('input size:', input.size())
        # print('target size:', target.size())
        # itr_count += 1
        input, target = input.cuda(), target.cuda()
        # print('input size  = ', input.size())
        # print('target size = ', target.size())
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()

        with torch.autograd.detect_anomaly():
            pred = model(input)  # @wx 注意输出
            loss = criterion(pred, target, size_average=True, batch_average=True)
            optimizer.zero_grad()
            loss.backward()  # compute gradient and do SGD step
            optimizer.step()

        # pred = model(input)  # @wx 注意输出
        # loss = criterion(pred, target, size_average=False, batch_average=True)
        # optimizer.zero_grad()
        # loss.backward()  # compute gradient and do SGD step
        # optimizer.step()

        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        pred = torch.argmax(pred, 1)
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'Loss={Loss:.5f} '
                  'IOU={result.iou:.2f}({average.iou:.2f}) '
                .format(
                epoch, i + 1, len(train_loader), data_time=data_time,
                gpu_time=gpu_time, Loss=loss.item(), result=result, average=average_meter.average()))

            current_step = epoch * batch_num + i
            logger.add_scalar('Train/Loss', loss, current_step)
            logger.add_scalar('Train/iou', result.iou, current_step)

    avg = average_meter.average()
    return avg
