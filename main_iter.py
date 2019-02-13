# -*- coding: utf-8 -*-
"""
 @Time    : 2019/2/13 16:42
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import os
import shutil
import socket
from datetime import datetime

import numpy as np

from tensorboardX import SummaryWriter

import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from torchvision.transforms import transforms
import dataloaders.transforms as tr
from libs import utils, criteria
from dataloaders.voc_aug import VOCAug

from libs.metrics import Result
from network.get_models import get_models

from train import train
from validation import validate


def parse_command():
    import argparse
    parser = argparse.ArgumentParser(description='DORN')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: ./run/run_1/checkpoint-5.pth.tar)')
    parser.add_argument('--model', default='deeplabv3plus', type=str, help='train which network')
    parser.add_argument('--crf', default=False, type=bool, help='if true, use crf as post process.')
    parser.add_argument('--msc', default=False, type=bool, help='if true, use multi-scale input.')
    parser.add_argument('--freeze', default=True, type=bool)
    parser.add_argument('--iter_size', default=2, type=int, help='when iter_size, opt step forward')
    parser.add_argument('-b', '--batch_size', default=8, type=int, help='mini-batch size (default: 4)')
    parser.add_argument('--iter_max', default=200, type=int, metavar='N',
                        help='number of max iter to run (default: 15)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate (default 0.0001)')
    parser.add_argument('--lr_patience', default=2, type=int,
                        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--dataset', default='vocaug', type=str,
                        help='dataset used for training, kitti and nyu is available')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--gpu', default=None, type=str, help='if not none, use Single GPU')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    args = parser.parse_args()
    return args


args = parse_command()
print(args)

# if setting gpu id, the using single GPU
if args.gpu:
    print('Single GPU Mode.')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

best_result = Result()
best_result.set_to_worst()


def create_loader(args):
    if args.dataset == 'vocaug':
        composed_transforms_tr = transforms.Compose([
            tr.RandomSized(512),
            tr.RandomRotate(15),
            tr.RandomHorizontalFlip(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        composed_transforms_ts = transforms.Compose([
            tr.FixedResize(size=(512, 512)),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        train_set = VOCAug(split='train', transform=composed_transforms_tr)
        val_set = VOCAug(split='val', transform=composed_transforms_ts)
    else:
        print('Database {} not available.'.format(args.dataset))
        raise NotImplementedError

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader


def main():
    global args, best_result, output_directory

    # set random seed
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        args.batch_size = args.batch_size * torch.cuda.device_count()
    else:
        print("Let's use GPU ", torch.cuda.current_device())

    train_loader, val_loader = create_loader(args)

    if args.resume:
        assert os.path.isfile(args.resume), \
            "=> no checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)

        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        optimizer = checkpoint['optimizer']

        # solve 'out of memory'
        model = checkpoint['model']

        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

        # clear memory
        del checkpoint
        # del model_dict
        torch.cuda.empty_cache()
    else:
        print("=> creating Model")
        model = get_models(args)
        print("=> model created.")
        start_epoch = 0

        # different modules have different learning rate
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        optimizer = torch.optim.SGD(train_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # You can use DataParallel() whether you use Multi-GPUs or not
        model = nn.DataParallel(model).cuda()