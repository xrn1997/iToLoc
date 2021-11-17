import os
import time

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tools import utils


class UJIndoorLocDataSet(Dataset):
    def __init__(self, dataset_path, train=True, noise=True):
        if train:
            train_path = "trainingData.csv"
        else:
            train_path = "validationData.csv"

        all_data = np.loadtxt(dataset_path + train_path, delimiter=',', skiprows=1, dtype=np.float32)
        self._data_len = all_data.shape[0]  # RSS指纹数量
        self.ap_len = all_data.shape[1] - 9  # AP数量 520
        self._rss = torch.from_numpy(all_data[:, :-9])  # RSS 520维向量
        self._co_data = all_data[:, -9:-6]  # 经度、纬度、楼层
        self._space = all_data[:, -6:-3]  # 楼、房间、相对位置（门内1、门外2）
        self._collector = all_data[:, -3:-1]  # 用户、手机
        self._date = all_data[:, -1:].reshape(self._data_len)  # 时间戳，如：1371713733
        # 生成位置标签
        if os.path.exists(dataset_path + "/co_dic.npy"):
            co_dic = np.load(dataset_path + "/co_dic.npy", allow_pickle=True).item()
        else:
            co_dic = {}  # 字典
        co_idx = 0  # 索引
        self._co_labels = []  # 位置标签
        for c in self._co_data:
            temp = c.tolist()
            if temp not in co_dic.values():
                co_dic[co_idx] = temp
                self._co_labels.append(co_idx)
                co_idx = co_idx + 1
            else:
                self._co_labels.append(list(co_dic.keys())[list(co_dic.values()).index(temp)])
        if noise:
            for n in range(len(self._co_labels)):
                self._co_labels[n] = utils.gen_labels(self._co_labels[n], types=len(co_dic))
        np.save(dataset_path + "/co_dic.npy", co_dic)
        np.save(dataset_path + "/co_labels.npy", self._co_labels)

        self.co_size = len(co_dic)  # 位置数量

        # 生成域标签
        self.date_domain = []
        for i in self._date:
            date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(i)))
            if int(date[-8:-6]) < 16:
                self.date_domain.append("A")
            elif int(date[-8:-6]) < 19:
                self.date_domain.append("B")
            else:
                self.date_domain.append("C")
        self.temp = pd.DataFrame(
            {'date': self.date_domain, 'device': [int(i[1]) for i in self._collector]})
        self._one_hot_label = pd.get_dummies(self.temp, columns=self.temp.columns)
        self.domain_size = self._one_hot_label.shape[1]  # 域的数量

    def __len__(self):
        return self._data_len

    def __getitem__(self, index):
        # 将1维RSS向量转换为2维RSS向量
        rss_item = self._rss[index].tolist()
        mx = np.matrix(rss_item * self.ap_len, dtype=np.float32).reshape(self.ap_len, self.ap_len).transpose()
        for i in range(0, self.ap_len):
            mx[:, i] = (mx[:, i] - rss_item[i]) / (rss_item[i] - 1)
        result = mx.A.reshape(1, self.ap_len, self.ap_len)  # 通道数为 1
        result = torch.from_numpy(result)

        return result, self._co_data[index], torch.Tensor(self._co_labels[index]), self._one_hot_label.values[index]


#  测试用代码
if __name__ == '__main__':
    uj_indoor_loc_path = "./UJIndoorLoc/"

    np.set_printoptions(threshold=np.inf)  # threshold 指定超过多少使用省略号，np.inf代表无限大
    dataset = UJIndoorLocDataSet(uj_indoor_loc_path, train=True)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=2,
                            num_workers=3,
                            pin_memory=True)
    print(len(dataset))
    print("ap_len", dataset.ap_len)
    print("co_size", dataset.co_size)
    print("domain_size", dataset.domain_size)
    for data in dataloader:
        data[1] *= torch.tensor([1, 1, 3])  # 假定每层楼高3米
        print(data[2])
        # coordinate = data[1] - data[1][data[2]]
        break
