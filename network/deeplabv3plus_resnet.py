# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/31 16:56
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import math

import torch.nn.functional as F
from torch.utils import model_zoo

from network.base.oprations import ASPP_module
from network.base.resnet import *


class DeeplabV3Plus(ResNet):

    def __init__(self, n_class, block, layers, pyramids, grids, output_stride=16):
        self.inplanes = 64
        super(DeeplabV3Plus, self).__init__()
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
        else:
            raise NotImplementedError

        # Backbone Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=grids, stride=strides[3], rate=rates[3])

        # Deeplab Modules
        self.aspp1 = ASPP_module(2048, 256, rate=pyramids[0])
        self.aspp2 = ASPP_module(2048, 256, rate=pyramids[1])
        self.aspp3 = ASPP_module(2048, 256, rate=pyramids[2])
        self.aspp4 = ASPP_module(2048, 256, rate=pyramids[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())

        #  the resulting feature of the deeplabv3 encoder
        self._conv1 = nn.Sequential(nn.Conv2d(1280, 256, kernel_size=1, stride=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())

        # adopt [1x1, 48] for channel reduction.
        self._conv2 = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU())

        self._conv3 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    # 3x3 conv to refine the features
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.Conv2d(256, n_class, kernel_size=1, stride=1))

        self.init_weight()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_features = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self._conv1(x)
        x = F.upsample(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)
        """
            above is the encoder
        """

        low_level_features = self._conv2(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self._conv3(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def get_1x_lr_params(self):
        b = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k

    def get_10x_lr_params(self):
        b = [self.aspp1, self.aspp2, self.aspp3, self.aspp4, self.global_avg_pool,
             self._conv1, self._conv2, self._conv3]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_backbone_bn(self):
        self.bn1.eval()

        for m in self.layer1:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for m in self.layer2:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for m in self.layer3:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for m in self.layer4:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def resnet101(n_class, output_stride=16, pretrained=True):
    if output_stride == 16:
        pyramids = [1, 6, 12, 18]
        grids = [1, 2, 4]
    elif output_stride == 8:
        pyramids = [1, 12, 24, 36]
        grids = [1, 2, 1]
    else:
        raise NotImplementedError

    model = DeeplabV3Plus(n_class=n_class, block=Bottleneck, layers=[3, 4, 23, 3],
                          pyramids=pyramids, grids=grids, output_stride=output_stride)

    if pretrained:
        pretrain_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
                # print(k)
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)

    return model


if __name__ == '__main__':
    model = resnet101(n_class=21, output_stride=16, pretrained=True)

    img = torch.randn(4, 3, 512, 512)

    with torch.no_grad():
        output = model.forward(img)

    print(output.size())
