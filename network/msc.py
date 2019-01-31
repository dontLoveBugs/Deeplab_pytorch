# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/31 19:03
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSC(nn.Module):
    """Multi-scale inputs"""

    def __init__(self, scale, pyramids=[0.5, 0.75]):
        super(MSC, self).__init__()
        self.scale = scale
        self.pyramids = pyramids

    def forward(self, x):
        # Original
        logits = self.scale(x)
        interp = lambda l: F.interpolate(
            l, size=logits.shape[2:], mode="bilinear", align_corners=True
        )

        # Scaled
        logits_pyramid = []
        for p in self.pyramids:
            size = [int(s * p) for s in x.shape[2:]]
            h = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
            logits_pyramid.append(self.scale(h))

        # Pixel-wise max
        logits_all = [logits] + [interp(l) for l in logits_pyramid]
        logits_max = torch.max(torch.stack(logits_all), dim=0)[0]

        if self.training:
            return [logits] + logits_pyramid + [logits_max]
        else:
            return logits_max

    def freeze_bn(self):
        self.scale.freeze_bn()

    def freeze_backbone_bn(self):
        self.scale.freeze_backbone_bn()

    def get_1x_lr_params(self):
        return self.scale.get_1x_lr_params()

    def get_10x_lr_params(self):
        return self.scale.get_10x_lr_params()
