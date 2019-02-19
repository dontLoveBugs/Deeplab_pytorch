# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/30 16:59
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""
import os
import shutil
import socket
import time
from datetime import datetime

import numpy as np

from tensorboardX import SummaryWriter

import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision.transforms import transforms
from tqdm import tqdm

import dataloaders.transforms as tr
from libs import utils, criteria
from dataloaders.voc_aug import VOCAug

from libs.metrics import Result, AverageMeter
from network.get_models import get_models

from validation import validate


def parse_command():
    import argparse
    parser = argparse.ArgumentParser(description='DORN')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: ./run/run_1/checkpoint-5.pth.tar)')
    parser.add_argument('--model', default='deeplabv2', type=str, help='train which network')
    parser.add_argument('--crf', default=False, type=bool, help='if true, use crf as post process.')
    parser.add_argument('--msc', default=False, type=bool, help='if true, use multi-scale input.')
    parser.add_argument('--freeze', default=True, type=bool)
    parser.add_argument('--iter_size', default=2, type=int, help='when iter_size, opt step forward')
    parser.add_argument('-b', '--batch_size', default=4, type=int, help='mini-batch size (default: 4)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
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
    args = parse_command()
    print(args)

    # if setting gpu id, the using single GPU
    if args.gpu:
        print('Single GPU Mode.')
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    best_result = Result()
    best_result.set_to_worst()

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

    # when training, use reduceLROnPlateau to reduce learning rate
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=args.lr_patience)

    # loss function
    criterion = criteria._CrossEntropyLoss2d(size_average=True, batch_average=True)

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

    start_iter = len(train_loader) * start_epoch + 1
    max_iter = len(train_loader) * (args.epochs - start_epoch + 1) + 1
    iter_save = len(train_loader)

    # train
    model.train()
    if args.freeze:
        model.module.freeze_backbone_bn()
    output_directory = utils.get_output_directory(args, check=True)

    average_meter = AverageMeter()
    train_meter = AverageMeter()

    for it in tqdm(range(start_iter, max_iter + 1), total=max_iter, leave=False, dynamic_ncols=True):
        optimizer.zero_grad()

        loss = 0

        data_time = 0
        gpu_time = 0

        for _ in range(args.iter_size):

            end = time.time()

            try:
                samples = next(loader_iter)
            except:
                loader_iter = iter(train_loader)
                samples = next(loader_iter)

            input = samples['image'].cuda()
            target = samples['label'].cuda()

            torch.cuda.synchronize()
            data_time_ = time.time()
            data_time += data_time_ - end

            with torch.autograd.detect_anomaly():
                preds = model(input)  # @wx 注意输出

                # print('#train preds size:', len(preds))
                # print('#train preds[0] size:', preds[0].size())
                iter_loss = 0
                if args.msc:
                    for pred in preds:
                        # Resize labels for {100%, 75%, 50%, Max} logits
                        target_ = utils.resize_labels(target, shape=(pred.size()[-2], pred.size()[-1]))
                        # print('#train pred size:', pred.size())
                        iter_loss += criterion(pred, target_)
                else:
                    pred = preds
                    target_ = utils.resize_labels(target, shape=(pred.size()[-2], pred.size()[-1]))
                    # print('#train pred size:', pred.size())
                    # print('#train target size:', target.size())
                    iter_loss += criterion(pred, target_)

                # Backpropagate (just compute gradients wrt the loss)
                iter_loss /= args.iter_size
                iter_loss.backward()

                loss += float(iter_loss)

            gpu_time += time.time() - data_time_

        torch.cuda.synchronize()

        # Update weights with accumulated gradients
        optimizer.step()

        # measure accuracy and record loss
        result = Result()
        pred = F.softmax(pred, dim=1)

        result.evaluate(pred.data.cpu().numpy(), target.data.cpu().numpy(), n_class=21)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        train_meter.update(result, gpu_time, data_time, input.size(0))

        if it % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Iter: [{0}/{1}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'Loss={Loss:.5f} '
                  'MeanAcc={result.mean_acc:.3f}({average.mean_acc:.3f}) '
                  'MIOU={result.mean_iou:.3f}({average.mean_iou:.3f}) '
                  .format(it, max_iter, data_time=data_time, gpu_time=gpu_time,
                          Loss=loss, result=result, average=average_meter.average()))
            logger.add_scalar('Train/Loss', loss, it)
            logger.add_scalar('Train/mean_acc', result.mean_iou, it)
            logger.add_scalar('Train/mean_iou', result.mean_acc, it)

        if it % iter_save == 0:
            epoch = it // iter_save
            resu1t, img_merge = validate(args, val_loader, model, epoch=epoch, logger=logger)

            # when rml doesn't fall, reduce learning rate
            scheduler.step(result.mean_iou)

            # save the change of learning_rate
            for i, param_group in enumerate(optimizer.param_groups):
                old_lr = float(param_group['lr'])
                logger.add_scalar('Lr/lr_' + str(i), old_lr, it)

            # vis the change between train and test
            train_avg = train_meter.average()
            logger.add_scalar('TrainVal/mean_acc', {'train_mean_acc':train_avg.mean_acc,
                                                    'test_mean_acc':result.mean_acc}, epoch)
            logger.add_scalar('TrainVal/mean_iou', {'train_mean_iou':train_avg.mean_iou,
                                                    'test_mean_iou':result.mean_iou}, epoch)
            train_meter.reset()
            # remember best rmse and save checkpoint
            is_best = result.mean_iou < best_result.mean_iou
            if is_best:
                best_result = result
                with open(best_txt, 'w') as txtfile:
                    txtfile.write(
                        "epoch={}, mean_iou={:.3f}, mean_acc={:.3f}"
                        "t_gpu={:.4f}".
                            format(epoch, result.mean_iou, result.mean_acc, result.gpu_time))
                if img_merge is not None:
                    img_filename = output_directory + '/comparison_best.png'
                    utils.save_image(img_merge, img_filename)

            # save checkpoint for each epoch
            utils.save_checkpoint({
                'args': args,
                'epoch': epoch,
                'model': model,
                'best_result': best_result,
                'optimizer': optimizer,
            }, is_best, it, output_directory)

            # change to train mode
            model.train()
            if args.freeze:
                model.module.freeze_backbone_bn()

    logger.close()


if __name__ == '__main__':
    main()
