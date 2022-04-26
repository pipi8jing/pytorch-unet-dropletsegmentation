# -*- coding:utf-8 -*-
# @Time : 2021/11/7 11:55
# @Author : Fang mirror
# @Site :
# @File : bufferedpatchdataset.py
# @Software: PyCharm
import numpy as np
import torch
from tqdm import tqdm
from fnet.data.fnetdataset import FnetDataset


class BufferedPatchDataset(FnetDataset):
    def __init__(self,
                 dataset,
                 patch_size,
                 buffer_size=30,
                 buffer_switch_frequency=1920,
                 npatches=100000,
                 verbose=False,
                 transform=None,
                 shuffle_images=True,
                 dim_squeeze=None):
        self.dataset = dataset
        self.buffer_switch_frequency = buffer_switch_frequency
        self.npatches = npatches
        self.verbose = verbose
        self.transform = transform
        self.shuffle_images = shuffle_images
        self.dim_squeeze = dim_squeeze

        # 读取buffer计数
        self.counter = 0
        # buffer存放
        self.buffer = list()
        # buffer索引历史
        self.buffer_history = list()

        # 数据集长度
        shuffle_data_rank = np.arange(0, len(self.dataset))

        # 打乱数据集
        if self.shuffle_images:
            np.random.shuffle(shuffle_data_rank)

        # 进度条，以buffer_size为行程
        pbar = tqdm(range(0, buffer_size))

        for i in pbar:
            if self.verbose:
                pbar.set_description("buffering images")

            # 从打乱的数据集中按顺序取数据索引
            data_ele_index = shuffle_data_rank[i]
            # 按索引取数据
            data_ele = self.dataset[data_ele_index]
            # 得到数据的shape信息
            data_ele_size = data_ele[0].size()
            # 将数据在数据集中的索引存放起来
            self.buffer_history.append(data_ele_index)
            # 将数据放到buffer内
            self.buffer.append(data_ele)

        # 由于数据集肯定多于buffer_size，未取的数据集索引存放至remaining_buffer_rank中
        self.remaining_buffer_rank = shuffle_data_rank[i + 1:]
        self.patch_size = [data_ele_size[0]] + patch_size

    def __len__(self):
        return self.npatches

    def __getitem__(self, index):
        self.counter += 1

        # 如果达到设定的要求，就从数据集中引入新的数据进buffer
        if (self.buffer_switch_frequency > 0) and (self.counter % self.buffer_switch_frequency == 0):
            if self.verbose:
                print("Inserting new item into buffer")

            self.insert_new_ele_into_buffer()

        return self.get_random_patch()

    def insert_new_ele_into_buffer(self):
        """
        从数据集中加载新的数据进入buffer
        :return:
        """
        # 抛弃第一个数据
        self.buffer.pop(0)

        if self.shuffle_images:
            # 如果数据集全部取完了，就重新开始遍历数据集
            if len(self.remaining_buffer_rank) == 0:
                self.remaining_buffer_rank = np.arange(0, len(self.dataset))
                np.random.shuffle(self.remaining_buffer_rank)

            new_data_ele_index = self.remaining_buffer_rank[0]
            self.remaining_buffer_rank = self.remaining_buffer_rank[1:]
        else:
            # 未打乱情况下，按顺序取
            new_data_ele_index = self.buffer_history[-1] + 1
            # 如果数据集取完了，从头开始取数据
            if new_data_ele_index == len(self.dataset):
                new_data_ele_index = 0

        self.buffer_history.append(new_data_ele_index)
        self.buffer.append(self.dataset[new_data_ele_index])

        if self.verbose:
            print("Added item {0}".format(new_data_ele_index))

    def get_random_patch(self):
        """
        从图像中得到随机patch
        :return:
        """
        # 从buffer中随机取一个数据
        buffer_index = np.random.randint(len(self.buffer))
        data_ele = self.buffer[buffer_index]

        # patch选取的开始点和结束点
        starts = np.array([np.random.randint(0, d - p + 1) if d - p + 1 >= 1 else 0
                           for d, p in zip(data_ele[0].size(), self.patch_size)])
        ends = starts + np.array(self.patch_size)

        # 利用slice函数在数据中取patch
        index = [slice(s, e) for s, e in zip(starts, ends)]
        patch = [d[tuple(index)] for d in data_ele]

        if self.dim_squeeze is not None:
            patch = [torch.squeeze(d, self.dim_squeeze) for d in patch]
        return patch

    def get_buffer_history(self):
        return self.buffer_history


