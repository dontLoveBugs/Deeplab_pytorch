# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/30 23:34
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import numpy as np
import torch
import time

import torch.nn.functional as F

from libs.DenseCRF import DenseCRF
from libs.metrics import AverageMeter, Result
from libs import utils


def validate(args, val_loader, model, epoch, logger):
    average_meter = AverageMeter()
    model.eval()  # switch to train mode

    output_directory = utils.get_output_directory(args, check=True)
    skip = len(val_loader) // 9  # save images every skip iters

    if args.crg:
        ITER_MAX = 10
        POS_W = 3
        POS_XY_STD = 1
        BI_W = 4
        BI_XY_STD = 67
        BI_RGB_STD = 3

        postprocessor = DenseCRF(
            iter_max=ITER_MAX,
            pos_xy_std=POS_XY_STD,
            pos_w=POS_W,
            bi_xy_std=BI_XY_STD,
            bi_rgb_std=BI_RGB_STD,
            bi_w=BI_W,
        )

    end = time.time()

    for i, samples in enumerate(val_loader):

        input = samples['image']
        target = samples['label']

        # itr_count += 1
        input, target = input.cuda(), target.cuda()
        # print('input size  = ', input.size())
        # print('target size = ', target.size())
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()

        with torch.no_grad():
            pred = model(input)  # @wx 注意输出

        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()

        pred = F.softmax(pred, 1)

        if pred.size() != target.size():
            pred = F.interpolate(pred, size=(target.size()[-2], target.size()[-1]), mode='bilinear', align_corners=True)

        pred = pred.data.cpu().numpy()
        target = target.data.cpu().numpy()

        # Post Processing
        if args.crf:
            images = input.numpy().astype(np.uint8).transpose(0, 2, 3, 1)
            pred = joblib.Parallel(n_jobs=-1)(
                [joblib.delayed(postprocessor)(*pair) for pair in zip(images, pred)]
            )

        result.evaluate(pred, target)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        rgb = input

        if i == 0:
            img_merge = utils.merge_into_row(rgb, target, pred)
        elif (i < 8 * skip) and (i % skip == 0):
            row = utils.merge_into_row(rgb, target, pred)
            img_merge = utils.add_row(img_merge, row)
        elif i == 8 * skip:
            filename = output_directory + '/comparison_' + str(epoch) + '.png'
            utils.save_image(img_merge, filename)

        if (i + 1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'IOU={result.iou:.2f}({average.iou:.2f})'.format(
                i + 1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()
    logger.add_scalar('Test/iou', avg.iou, epoch)

    print('\n*\n'
          'IOU={average.iou:.3f}\n'
          't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    return avg, img_merge
