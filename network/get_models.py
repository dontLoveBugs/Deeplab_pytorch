# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/30 23:19
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

from network import deeplab, deeplabv2, deeplabv3, deeplabv3plus


def get_models(args):
    if args.model == 'deeplabv3':
        return deeplabv3.DeepLabv3(nInputChannels=3, n_classes=21, os=16, pretrained=True)
    elif args.model == 'deeplabv3plus':
        return deeplabv3plus.DeepLabv3_plus(nInputChannels=3, n_classes=21, os=16, pretrained=True).cuda()
    else:
        print('Model {} not implemented.'.format(args.model))
        raise NotImplementedError
