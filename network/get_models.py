# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/30 23:19
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

from network import deeplab, deeplabv2, deeplabv3, deeplabv3plus_resnet
from network.msc import MSC


def get_models(args):
    if args.model == 'deeplabv2':
        if args.msc:
            return MSC(scale= deeplabv2.resnet101(n_class=21, pretrained=True),  pyramids=[0.5, 0.75])
        return deeplabv2.resnet101(n_class=21, pretrained=True)
    elif args.model == 'deeplabv3':
        if args.msc:
            return MSC(scale= deeplabv3.resnet101(n_class=21, output_stride=16, pretrained=True),
                       pyramids=[0.5, 0.75])
        return deeplabv3.resnet101(n_class=21, output_stride=16, pretrained=True)
    elif args.model == 'deeplabv3plus':
        if args.msc:
            return MSC(scale= deeplabv3plus_resnet.resnet101(n_class=21, output_stride=16, pretrained=True),
                       pyramids=[0.5, 0.75])
        return deeplabv3plus_resnet.resnet101(n_class=21, output_stride=16, pretrained=True)
    else:
        print('Model {} not implemented.'.format(args.model))
        raise NotImplementedError
