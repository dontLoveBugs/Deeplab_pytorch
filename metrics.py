# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/30 23:38
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import torch
import math
import numpy as np

import utils


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)


class Result(object):
    def __init__(self):
        self.iou = 0
        self.data_time, self.gpu_time = 0, 0

    def set_to_worst(self):
        self.iou = np.inf
        self.data_time, self.gpu_time = 0, 0

    def update(self, iou, gpu_time, data_time):
        self.iou = iou
        self.data_time, self.gpu_time = data_time, gpu_time

    def evaluate(self, output, target):
        self.iou = utils.get_iou(output, target, n_classes=21)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_iou = 0
        self.sum_data_time, self.sum_gpu_time = 0, 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n

        self.sum_iou += n * result.iou
        self.sum_data_time += n * data_time
        self.sum_gpu_time += n * gpu_time

    def average(self):
        avg = Result()
        avg.update(
            self.sum_iou / self.count,
            self.sum_gpu_time / self.count,
            self.sum_data_time / self.count)
        return avg
