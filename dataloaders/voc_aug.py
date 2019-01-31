# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/30 19:29
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import os
from PIL import Image
from torch.utils.data import Dataset

from dataloaders.path import Path


class VOCAug(Dataset):

    def __init__(self, base_dir=Path.db_root_dir('vocaug'),
                 split='train',
                 transform=None):
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'img')
        self._cat_dir = os.path.join(self._base_dir, 'gt')
        self._list_dir = os.path.join(self._base_dir, 'list')

        self.transform = transform

        # print(self._base_dir)

        if split == 'train':
            list_path = os.path.join(self._list_dir, 'train_aug.txt')
        elif split == 'val':
            list_path = os.path.join(self._list_dir, 'val.txt')
        else:
            print('error in split:', split)
            exit(-1)

        self.filenames = [i_id.strip() for i_id in open(list_path)]

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.filenames)))

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}
        # print('test!!!!')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.filenames)

    def _make_img_gt_point_pair(self, index):

        filename = self.filenames[index]
        # print('filename = ', filename)

        _img = Image.open(self._image_dir + "/" + str(filename) + '.jpg').convert('RGB')
        _target = Image.open(self._cat_dir + "/" + str(filename) + '.png')

        return _img, _target

    def __str__(self):
        return 'VOCAug(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders import transforms as tr
    from libs.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import numpy as np

    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomSized(512),
        tr.RandomRotate(15),
        tr.ToTensor()])

    voc_train = VOCAug(split='train', transform=composed_transforms_tr)

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=1)

    print(len(dataloader))

    for ii, sample in enumerate(dataloader):
        print(sample['image'].size())
        img = sample['image'].numpy()
        gt = sample['label'].numpy()

        for jj in range(sample["image"].size()[0]):
            tmp = np.array(gt[jj]).astype(np.uint8)
            print('#1 gt ', tmp.shape)
            tmp = np.squeeze(tmp, axis=0)
            print('#2 gt ', tmp.shape)
            segmap = decode_segmap(tmp, dataset='vocaug')
            print('#1 im ', img[jj].shape)
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0]).astype(np.uint8)
            print('#2 im ', img_tmp.shape)
            # plt.figure()
            # plt.title('display')
            # plt.subplot(211)
            # plt.imshow(img_tmp)
            # plt.subplot(212)
            # plt.imshow(segmap)

        if ii == 1:
            break
    plt.show(block=True)