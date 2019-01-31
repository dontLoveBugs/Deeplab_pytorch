# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/30 23:17
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=False)
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(),
                                        ignore_index=ignore_index, size_average=False)
    loss = criterion(logit, target.long())

    if size_average:
        loss /= (h * w)

    if batch_average:
        loss /= n

    return loss


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class _CrossEntropyLoss2d(nn.Module):
    def __init__(self, ignore_index=255, weight=None, size_average=True, batch_average=True):
        super(_CrossEntropyLoss2d, self).__init__()

        if weight is None:
            self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=False)
        else:
            self.loss = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(),
                                            ignore_index=ignore_index, size_average=False)

        self.size_avgrage = size_average
        self.batch_avgrage = batch_average

    def forward(self, logit, target):

        N, C, H, W = logit.size()

        target = target.squeeze(1)

        loss = self.loss(logit, target.long())

        if self.size_avgrage:
            loss /= (H * W)

        if self.batch_avgrage:
            loss /= N

        return loss
