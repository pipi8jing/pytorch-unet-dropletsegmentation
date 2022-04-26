# -*- coding:utf-8 -*-
# @Time : 2021/11/6 16:45
# @Author : Fang mirror
# @Site :
# @File : functions.py
# @Software: PyCharm
import importlib
import os
import fnet
import pandas as pd


def load_model(path_model, gpu_ids=0, module='fnet_model'):
    module_fnet_model = importlib.import_module('fnet.' + module)
    if os.path.isdir(path_model):
        path_model = os.path.join(path_model, 'model.p')
    model = module_fnet_model.Model()
    model.load_state(path_model, gpu_ids=gpu_ids)
    return model


def load_model_from_dir(path_model_dir, gpu_ids=0):
    """
    从指定文件夹下加载模型
    :param path_model_dir: 模型所在文件夹
    :param gpu_ids: gpu序号
    :return: 返回模型
    """
    assert os.path.isdir(path_model_dir)
    path_model_state = os.path.join(path_model_dir, 'model.p')
    model = fnet.fnet_model.Model()
    model.load_state(path_model_state, gpu_ids=gpu_ids)
    return model


class FnetLogger(object):
    """Log values in a dict of lists."""
    def __init__(self, path_csv=None, columns=None):
        if path_csv is not None:
            df = pd.read_csv(path_csv)
            self.columns = list(df.columns)
            self.data = df.to_dict(orient='list')
        else:
            self.columns = columns
            self.data = {}
            for c in columns:
                self.data[c] = []

    def __repr__(self):
        return 'FnetLogger({})'.format(self.columns)

    def add(self, entry):
        if isinstance(entry, dict):
            for key, value in entry.items():
                self.data[key].append(value)
        else:
            assert len(entry) == len(self.columns)
            for i, value in enumerate(entry):
                self.data[self.columns[i]].append(value)

    def to_csv(self, path_csv):
        """
        保存到csv文件方法
        :param path_csv: 文件路径
        :return:
        """
        dirname = os.path.dirname(path_csv)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        pd.DataFrame(self.data)[self.columns].to_csv(path_csv, index=False)
