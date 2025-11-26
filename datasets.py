import os, sys, random
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler


class TrainDataset_Com(torch.utils.data.Dataset):
    """
    用于加载“完整”数据集 (X_com, Y_com)，通常用于预训练
    """

    def __init__(self, X_list, Y_list):
        self.X_list = X_list
        self.Y_list = Y_list
        self.view_size = len(X_list)

    def __getitem__(self, index):
        current_x_list = []
        current_y_list = []
        for v in range(self.view_size):
            current_x = self.X_list[v][index]
            current_x_list.append(current_x)
            current_y = self.Y_list[v][index]
            current_y_list.append(current_y)
        return current_x_list, current_y_list

    def __len__(self):
        return self.X_list[0].shape[0]


class TrainDataset_All(torch.utils.data.Dataset):
    """
    用于加载 *所有* 数据 (X, Y) 及其缺失掩码 (Miss_list)
    """

    def __init__(self, X_list, Y_list, Miss_list):
        self.X_list = X_list
        self.Y_list = Y_list
        self.Miss_list = Miss_list  # 这是一个 (N, V) 的Numpy数组
        self.view_size = len(X_list)

    def __getitem__(self, index):
        current_x_list = []
        current_y_list = []
        current_miss_list = []  # (我们返回一个 (V,) 的掩码)

        # --- (这是核心修正) ---
        # 原始代码是 self.Miss_list[v][index]，这是 (v, idx) 索引
        # 我们的 Miss_list 是 (N, V) 形状，所以应该用 (idx, v) 索引

        current_miss_row = self.Miss_list[index]  # 直接获取第 index 行，形状为 (V,)

        for v in range(self.view_size):
            current_x = self.X_list[v][index]
            current_x_list.append(current_x)

            current_y = self.Y_list[v][index]
            current_y_list.append(current_y)

            # 从 (V,) 的行中获取第 v 个值
            current_miss = current_miss_row[v]
            current_miss_list.append(current_miss)

        # 将掩码列表转换为一个 (V,) 的张量
        current_miss_tensor = torch.tensor(current_miss_list, dtype=torch.float32)

        return current_x_list, current_y_list, current_miss_tensor
        # --- (修正结束) ---

    def __len__(self):
        return self.X_list[0].shape[0]


class Data_Sampler(object):
    """
    (保留) 这是一个自定义的采样器，虽然 PyTorch 的 DataLoader 更好，
    但如果您的训练代码依赖它，我们可以保留它。
    """

    def __init__(self, pairs, shuffle=False, batch_size=1, drop_last=False):
        if shuffle:
            self.sampler = RandomSampler(pairs)
        else:
            self.sampler = SequentialSampler(pairs)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batch = [batch]
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            batch = [batch]
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size