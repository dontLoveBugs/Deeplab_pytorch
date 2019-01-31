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


def validate(args, val_loader, model, epoch, logger):
    average_meter = AverageMeter()
    model.eval()  # switch to train mode

    output_directory = utils.get_output_directory(args, check=True)
    skip = len(val_loader) // 9  # save images every skip iters

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

        # print('#val pred:', pred.size())
        pred = torch.argmax(pred, 1)
        # print('#val #2 pred:', pred.size())
        result.evaluate(pred.data, target.data)
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
                  'RMSE={result.iou:.2f}({average.iou:.2f})'.format(
                i + 1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()
    logger.add_scalar('Test/iou', avg.iou, epoch)
    return avg, img_merge
