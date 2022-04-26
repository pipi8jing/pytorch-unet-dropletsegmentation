# -*- coding:utf-8 -*-
# @Time : 2021/11/6 15:43
# @Author : Fang mirror
# @Site :
# @File : fnetdataset.py.py
# @Software: PyCharm
import torch.utils.data
import typing


class FnetDataset(torch.utils.data.Dataset):
    def get_information(self, index) -> typing.Union[dict, str]:
        """Returns information to identify dataset element specified by index."""
        raise NotImplementedError

    def apply_transforms(self, pytorch_tensor):
        for t in self.transforms:
            pytorch_tensor = t(pytorch_tensor)

        return pytorch_tensor
