# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/30 23:19
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

from network import deeplab, deeplabv2, deeplabv3, deeplabv3plus_resnet


def get_models(args):
    if args.model == 'deeplabv3':
        return deeplabv3.resnet101(n_class=21, output_stride=16, pretrained=True)
    elif args.model == 'deeplabv3plus':
        return deeplabv3plus_resnet.resnet101(n_class=21, output_stride=16, pretrained=True)
    else:
        print('Model {} not implemented.'.format(args.model))
        raise NotImplementedError
