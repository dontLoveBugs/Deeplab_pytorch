# -*- coding: utf-8 -*-
"""
 @Time    : 2019/2/17 22:15
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


def fixed_padding(inputs, kernel_size, dilation):
    """
    https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/xception.py
    :param kernel_size:
    :param dilation:
    :return:
    """
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, [pad_beg, pad_end, pad_beg, pad_end])
    return padded_inputs


# from https://github.com/quark0/darts/blob/master/cnn/operations.py
class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Cell(nn.Module):
    def __init__(self, in_channels_h1, in_channels_h2, out_channels, dilation=1, activation=nn.ReLU6,
                 bn=nn.BatchNorm2d):
        """
        Initialization of inverted residual block
        :param in_channels_h1: number of input channels in h-1
        :param in_channels_h2: number of input channels in h-2
        :param out_channels: number of output channels
        :param t: the expansion factor of block
        :param s: stride of the first convolution
        :param dilation: dilation rate of 3*3 depthwise conv ?? fixme
        """
        super(Cell, self).__init__()
        self.in_ = in_channels_h1
        self.out_ = out_channels
        self.activation = activation

        if in_channels_h1 > in_channels_h2:
            self.preprocess = FactorizedReduce(in_channels_h2, in_channels_h1)
        elif in_channels_h1 < in_channels_h2:
            # todo check this
            self.preprocess = nn.ConvTranspose2d(in_channels_h2, in_channels_h1, 3, stride=2, padding=1, output_padding=1)
        else:
            self.preprocess = None

        #self.atr3x3 = DilConv(in_channels_h1, out_channels, 3, 1, 1, dilation)
        #self.atr5x5 = DilConv(in_channels_h1, out_channels, 5, 1, 2, dilation)

        #self.sep3x3 = SepConv(in_channels_h1, out_channels, 3, 1, 1)
        #self.sep5x5 = SepConv(in_channels_h1, out_channels, 5, 1, 2)

        # Top 1
        self.top1_atr5x5 = DilConv(in_channels_h1, in_channels_h1, 5, 1, 2, dilation)
        self.top1_sep3x3 = SepConv(in_channels_h1, in_channels_h1, 3, 1, 1)

        # Top 2
        self.top2_sep5x5_1 = SepConv(in_channels_h1, in_channels_h1, 5, 1, 2)
        self.top2_sep5x5_2 = SepConv(in_channels_h1, in_channels_h1, 5, 1, 2)

        # Middle
        self.middle_sep3x3_1 = SepConv(in_channels_h1, in_channels_h1, 3, 1, 1)
        self.middle_sep3x3_2 = SepConv(in_channels_h1, in_channels_h1, 3, 1, 1)

        # Bottom 1
        self.bottom1_atr3x3 = DilConv(in_channels_h1, in_channels_h1, 3, 1, 1, dilation)
        self.bottom1_sep3x3 = SepConv(in_channels_h1, in_channels_h1, 3, 1, 1)

        # Bottom 2
        self.bottom2_atr5x5 = DilConv(in_channels_h1, in_channels_h1, 5, 1, 2, dilation)
        self.bottom2_sep5x5 = SepConv(in_channels_h1, in_channels_h1, 5, 1, 2)

        self.concate_conv = nn.Conv2d(in_channels_h1*5, out_channels, 1)

    def forward(self, h_1, h_2):
        """
        :param h_1:
        :param h_2:
        :return:
        """

        if self.preprocess is not None:
            h_2 = self.preprocess(h_2)

        top1 = self.top1_atr5x5(h_2) + self.top1_sep3x3(h_1)
        bottom1 = self.bottom1_atr3x3(h_1) + self.bottom1_sep3x3(h_2)
        middle = self.middle_sep3x3_1(h_2) + self.middle_sep3x3_2(bottom1)

        top2 = self.top2_sep5x5_1(top1) + self.top2_sep5x5_2(middle)
        bottom2 = self.bottom2_atr5x5(top2) + self.bottom2_sep5x5(bottom1)

        concat = torch.cat([top1, top2, middle, bottom2, bottom1], dim=1)

        return self.concate_conv(concat)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, paddings, dilations):
        # todo depthwise separable conv
        super(ASPP, self).__init__()
        self.conv11 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False, ),
                                    nn.BatchNorm2d(256))
        self.conv33_1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
                                                padding=paddings[0], dilation=dilations[0], bias=False, ),
                                      nn.BatchNorm2d(256))
        self.conv33_2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
                                                padding=paddings[1], dilation=dilations[1], bias=False, ),
                                      nn.BatchNorm2d(256))
        self.conv33_3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
                                                padding=paddings[2], dilation=dilations[2], bias=False, ),
                                      nn.BatchNorm2d(256))
        self.concate_conv = nn.Sequential(nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
                                          nn.BatchNorm2d(256))
        # self.upsample = nn.Upsample(mode='bilinear', align_corners=True)

    def forward(self, x):
        conv11 = self.conv11(x)
        conv33_1 = self.conv33_1(x)
        conv33_2 = self.conv33_2(x)
        conv33_3 = self.conv33_3(x)

        # image pool and upsample
        image_pool = nn.AvgPool2d(kernel_size=x.size()[2:])
        image_pool = image_pool(x)
        image_pool = self.conv11(image_pool)
        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)
        upsample = upsample(image_pool)

        # concate
        concate = torch.cat([conv11, conv33_1, conv33_2, conv33_3, upsample], dim=1)

        return self.concate_conv(concate)


# Based on quark0/darts on github
class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        padded = F.pad(x, (0, 1, 0, 1), "constant", 0)
        path2 = self.conv_2(padded[:, :, 1:, 1:])
        out = torch.cat([self.conv_1(x), path2], dim=1)
        out = self.bn(out)
        return out