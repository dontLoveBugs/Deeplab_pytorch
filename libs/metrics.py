# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/30 23:38
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import torch
import math
import numpy as np

from libs import utils


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)


"""
  The two function named as _fast_hist, scores is Originally written by wkentaro
  https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
"""


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    # cls_iu = dict(zip(range(n_class), iu))

    # print('# type:', type(mean_iu), type(acc_cls), type(acc), type(fwavacc))

    return mean_iu, acc_cls, acc, fwavacc


class Result(object):
    def __init__(self):
        self.mean_iou = 0
        self.mean_acc = 0
        self.overall_acc = 0
        self.freqw_acc = 0
        self.data_time, self.gpu_time = 0, 0

    def set_to_worst(self):
        self.mean_iou = np.inf
        self.mean_acc = np.inf
        self.overall_acc = np.inf
        self.freqw_acc = np.inf
        self.data_time, self.gpu_time = 0, 0

    def update(self, mean_iou, mean_acc, overall_acc, freqw_accc, gpu_time, data_time):
        self.mean_iou = mean_iou
        self.mean_acc, self.overall_acc, self.freqw_acc = mean_acc, overall_acc, freqw_accc
        self.data_time, self.gpu_time = data_time, gpu_time

    def evaluate(self, output, target, n_class):
        output = np.argmax(output, axis=1)
        # print('output size:', output.shape)
        # print('target size:', target.shape)

        outputs = []
        targets = []
        outputs.append(output)
        targets.append(target)
        self.mean_iou, self.mean_acc, self.overall_acc, self.freqw_acc \
            = scores(targets, outputs, n_class=n_class)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_mean_iou = 0
        self.sum_mean_acc = 0
        self.sum_overall_acc = 0
        self.sum_freqw_acc = 0
        self.sum_data_time, self.sum_gpu_time = 0, 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n

        self.sum_mean_iou += result.mean_iou * n
        self.sum_mean_acc += result.mean_acc * n
        self.sum_overall_acc += result.overall_acc * n
        self.sum_freqw_acc += result.freqw_acc * n

        self.sum_data_time += n * data_time
        self.sum_gpu_time += n * gpu_time

    def average(self):
        avg = Result()
        avg.update(
            self.sum_mean_iou / self.count,
            self.sum_mean_acc / self.count,
            self.sum_overall_acc / self.count,
            self.sum_freqw_acc / self.count,
            self.sum_gpu_time / self.count,
            self.sum_data_time / self.count)
        return avg
