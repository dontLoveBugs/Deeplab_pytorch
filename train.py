# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/30 23:34
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import numpy as np

import torch
import torch.nn.functional as F
import time

from libs.metrics import AverageMeter, Result
from libs import utils


def train(args, train_loader, model, criterion, optimizer, epoch, logger):
    average_meter = AverageMeter()
    model.train()  # switch to train mode
    batch_num = len(train_loader)

    output_directory = utils.get_output_directory(args, check=True)

    end = time.time()

    iter_count = 0

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
            preds = model(input)  # @wx 注意输出

            loss = 0
            if args.multi_scale:
                for pred in preds:
                    # Resize labels for {100%, 75%, 50%, Max} logits
                    target_ = utils.resize_labels(target, shape=(pred.size()[-2], pred.size()[-1]))
                    print('#train pred size:', pred.size())
                    loss += criterion(pred, target_)
            else:
                pred = preds
                target_ = utils.resize_labels(target, shape=(pred.size()[-2], pred.size()[-1]))
                print('#train pred size:', pred.size())
                print('#train target size:', target.size())
                loss += criterion(pred, target_)

            # Backpropagate (just compute gradients wrt the loss)
            loss /= args.iter_size
            loss.backward()
            iter_count += 1

            if iter_count % args.iter_size:
                optimizer.step()
                optimizer.zero_grad()
                iter_count = 0

        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        pred = F.softmax(pred, dim=1)

        result.evaluate(pred.data.cpu().numpy(), target.data.cpu().numpy(), n_class=21)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'Loss={Loss:.5f} '
                  'MeanAcc={result.mean_acc:.3f}({average.mean_acc}) '
                  'MIOU={result.mean_iou:.2f}({average.mean_iou:.2f}) '
                .format(
                epoch, i + 1, len(train_loader), data_time=data_time,
                gpu_time=gpu_time, Loss=loss.item(), result=result, average=average_meter.average()))

            current_step = epoch * batch_num + i
            logger.add_scalar('Train/Loss', loss, current_step)
            logger.add_scalar('Train/mean_acc', result.mean_iou, current_step)
            logger.add_scalar('Train/mean_iou', result.mean_acc, current_step)

    avg = average_meter.average()
    return avg
