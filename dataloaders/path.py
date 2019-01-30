# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/30 19:30
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return '/home/data/model/wangxin/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif database == 'vocaug':
            return '/home/data/model/wangxin/VOCAug/'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError