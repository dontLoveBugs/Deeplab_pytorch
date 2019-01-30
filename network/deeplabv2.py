# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/30 16:59
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from network.resnet import *
from network.aspp_module import *


class DeepLabv2(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=21, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv2 model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv2, self).__init__()

        # Atrous Conv
        self.resnet_features = resnet101(nInputChannels, os, pretrained=pretrained)