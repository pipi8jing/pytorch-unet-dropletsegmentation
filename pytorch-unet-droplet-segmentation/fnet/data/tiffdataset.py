import os.path
import numpy as np
import tifffile
import torch
import fnet.transforms as transforms
from fnet.data.fnetdataset import FnetDataset


class TiffDataset(FnetDataset):
    """
    读取训练数据-->配对的明场数据和荧光数据
    """
    def __init__(self, img_path: str=None, flag=True,
                 transform_signal=[transforms.normalize],
                 transform_target=None):
        # 数据集文件夹路径
        self.signal_dir = os.path.join(img_path, '')
        self.target_dir = os.path.join(img_path, '')
        if not flag:
            self.signal_dir = os.path.join(img_path, '')
            self.target_dir = os.path.join(img_path, '')
        # 数据集所有文件名
        self.signal_path_list = os.listdir(self.signal_dir)
        self.target_path_list = os.listdir(self.target_dir)
        # 图像变换
        self.transform_signal = transform_signal
        self.transform_target = transform_target

    def __getitem__(self, index):
        im_out = []
        # 读取数据
        signal = tifffile.imread(os.path.join(self.signal_dir, self.signal_path_list[index]))
        target = tifffile.imread(os.path.join(self.target_dir, self.target_path_list[index]))
        # 图像按预期顺序进行变换
        for t in self.transform_signal:
            signal = t(signal)
            #signal[1] = t(signal[1])
        # 二维数据成三通道，三维数据成四通道
        signal = torch.tensor(signal[np.newaxis].astype(np.float32), dtype=torch.float32)
        im_out.append(signal)

        if self.transform_target is not None and (len(target) > 1):
            for t in self.transform_target:
                target = t(target)
            target = torch.tensor(target[np.newaxis].astype(np.float32), dtype=torch.float32)
            im_out.append(target)
        return im_out

    def __len__(self):
        return len(self.signal_path_list)


class TiffDatasetTest(FnetDataset):
    """
    读取测试数据-->仅读取明场数据，只能用于predict情况
    """
    def __init__(self, img_path: str=None, transform_signal=[transforms.normalize]):
        self.signal_dir = img_path
        self.signal_path_list = os.listdir(self.signal_dir)
        self.transform_signal = transform_signal

    def __getitem__(self, index):
        im_out = []
        signal = tifffile.imread(os.path.join(self.signal_dir, self.signal_path_list[index]))
        for t in self.transform_signal:
            signal = t(signal)
            #signal[1] = t(signal[1])
        signal = torch.tensor(signal[np.newaxis].astype(np.float32), dtype=torch.float32)
        im_out.append(signal)

        return im_out

    def __len__(self):
        return len(self.signal_path_list)

