# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/30 16:59
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""
import glob
import os
import numpy as np

import torch.data.Dataloader
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from torchvision.transforms import transforms
import dataloaders.transforms as tr
from dataloaders.voc_aug import VOCAug

import criteria


def parse_command():
    import argparse
    parser = argparse.ArgumentParser(description='DORN')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: ./run/run_1/checkpoint-5.pth.tar)')
    parser.add_argument('--model', default='deeplabv3plus', type=str, help='train which network')
    parser.add_argument('-b', '--batch-size', default=6, type=int, help='mini-batch size (default: 4)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
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
    parser.add_argument('--gpu', default='1', type=str, help='if not none, use Single GPU')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    args = parser.parse_args()
    return args


args = parse_command()
print(args)


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

        train_set = VOCAug.VOCAug(split='train', transform=composed_transforms_tr)
        val_set = VOCAug.VOCAug(split='val', transform=composed_transforms_ts)
    else:
        print('Database {} not available.'.format(args.dataset))
        raise NotImplementedError

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader


def get_output_directory(args):
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    save_dir_root = os.path.join(save_dir_root, 'result', args.model, args.dataset)
    if args.resume:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) if runs else 0
    else:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))
    return save_dir


def main(args):
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
        model = get_models(args.dataset)
        print("=> model created.")
        start_epoch = 0

        # different modules have different learning rate
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        optimizer = torch.optim.SGD(train_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # You can use DataParallel() whether you use Multi-GPUs or not
        model = nn.DataParallel(model).cuda()

    # when training, use reduceLROnPlateau to reduce learning rate
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=args.lr_patience)

    # loss function
    criterion = criteria

    # create directory path
    output_directory = utils.get_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    best_txt = os.path.join(output_directory, 'best.txt')
    config_txt = os.path.join(output_directory, 'config.txt')

    # write training parameters to config file
    if not os.path.exists(config_txt):
        with open(config_txt, 'w') as txtfile:
            args_ = vars(args)
            args_str = ''
            for k, v in args_.items():
                args_str = args_str + str(k) + ':' + str(v) + ',\t\n'
            txtfile.write(args_str)

    # create log
    log_path = os.path.join(output_directory, 'logs',
                            datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    logger = SummaryWriter(log_path)


if __name__ == '__main__':
    main(args)
