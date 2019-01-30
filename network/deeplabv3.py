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


class DeepLabv3(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=21, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3 model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3, self).__init__()

        # Atrous Conv
        self.resnet_features = resnet101(nInputChannels, os, pretrained=pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, kernel_size=1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())

        # get result features from the concat
        self.conv1 = nn.Sequential(nn.Conv2d(1280, 256, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())

        # generate the final logits
        self.conv2 = nn.Conv2d(256, n_classes, 1, bias=False)

    def forward(self, input):
        x, _ = self.resnet_features(input)

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        # image-level features
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_1x_lr_params(self):
        b = [self.resnet_features]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k

    def get_10x_lr_params(self):
        b = [self.aspp1, self.aspp2, self.aspp3, self.aspp4, self.conv1, self.conv2]
        for j in range(len(b)):
            for k in b[j].parameters():
                if k.requires_grad:
                    yield k


if __name__ == '__main__':
    img = torch.randn(3, 3, 513, 513).cuda()
    model = DeepLabv3(nInputChannels=3, n_classes=21, os=16, pretrained=True).cuda()

    with torch.no_grad():
        pred = model.forward(img)

    print(pred.size())
