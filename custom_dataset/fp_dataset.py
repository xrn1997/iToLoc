import os
import time

import numpy as np
import pandas as pd
import torch
from logzero import logger
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from tools import utils


class UJIndoorLocDataSet(Dataset):
    def __init__(self, dataset_path, train=True, noise=False):
        if train:
            file_name = "trainingData.csv"
        else:
            file_name = "validationData.csv"

        all_data = np.loadtxt(dataset_path + file_name, delimiter=',', skiprows=1, dtype=np.float32)
        self._data_len = all_data.shape[0]  # RSS指纹数量
        self.ap_len = all_data.shape[1] - 9  # AP数量 520
        self._rss = torch.from_numpy(all_data[:, :-9])  # RSS 520维向量
        self._co_data = all_data[:, -9:-6]  # 经度、纬度、楼层
        self._space = all_data[:, -6:-3]  # 楼、房间、相对位置（门内1、门外2）
        self._collector = all_data[:, -3:-1]  # 用户、手机
        self._date = all_data[:, -1:].reshape(self._data_len)  # 时间戳，如：1371713733

        # 加载位置标签字典
        if os.path.exists(dataset_path + "/co_dic.npy"):
            co_dic = np.load(dataset_path + "/co_dic.npy", allow_pickle=True).item()
        elif not train:
            raise Exception("找不到位置标签字典，无法生成验证集的位置标签")
        else:
            co_dic = {}  # 字典

        self._co_labels = []  # 位置标签
        if train:
            # 生成位置标签字典和训练集的位置标签
            co_idx = 0  # 索引
            for d in self._co_data:
                temp = d.tolist()
                if temp not in co_dic.values():
                    co_dic[co_idx] = temp
                    self._co_labels.append(co_idx)
                    co_idx = co_idx + 1
                else:
                    self._co_labels.append(list(co_dic.keys())[list(co_dic.values()).index(temp)])
            np.save(dataset_path + "/co_dic.npy", co_dic)
        else:
            # 根据位置标签字典生成验证集的位置标签
            for d in self._co_data:
                temp = 100
                idx = -1
                for cd in co_dic.values():
                    if d[2] == cd[2]:
                        distance = torch.pairwise_distance(torch.Tensor(d), torch.Tensor(cd), eps=0).item()
                        if temp > distance:
                            temp = distance
                            idx = list(co_dic.keys())[list(co_dic.values()).index(cd)]
                if idx == -1:
                    raise Exception("存在脏数据: " + str(d))
                else:
                    self._co_labels.append(idx)

        self.co_size = len(co_dic)  # 位置字典大小，933

        if train and noise:
            # 只有训练集可以添加噪声，这里默认是将每一个标签转换成三个添加噪声的933维的one-hot向量。
            if os.path.exists(dataset_path + "/co_noise" + str(train) + ".npy"):
                noise_list = np.load(dataset_path + "/co_noise" + str(train) + ".npy", allow_pickle=True)
            else:
                noise_list = []
                for i in range(len(self._co_labels)):
                    noise_list.append(utils.gen_noise(types=self.co_size))
                np.save(dataset_path + "/co_noise" + str(train) + ".npy", noise_list)
            temp = []
            for idx, cl in enumerate(self._co_labels):
                temp.append(torch.Tensor(utils.gen_labels(label=cl, noise_list=noise_list[idx])))
            self._co_labels = temp

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
            # mx[:, i] = -(abs(mx[:, i]) - abs(rss_item[i]))
        result = mx.A.reshape(1, self.ap_len, self.ap_len)  # 通道数为 1
        result = torch.from_numpy(result)
        return result, self._co_data[index], self._co_labels[index], self._one_hot_label.values[index]


#  测试用代码
if __name__ == '__main__':
    uj_indoor_loc_path = "./UJIndoorLoc/"
    try:
        dataset = UJIndoorLocDataSet(uj_indoor_loc_path, train=False)
    except Exception as ep:
        logger.error(ep)
    else:
        dataloader = DataLoader(dataset=dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=True)
        logger.debug(len(dataset))
        logger.debug("ap_len: " + str(dataset.ap_len))
        logger.debug("co_size: " + str(dataset.co_size))
        logger.debug("domain_size: " + str(dataset.domain_size))
        for data in dataloader:
            logger.debug(data[2])
            break
            # data[1] *= torch.tensor([1, 1, 3])  # 假定每层楼高3米
            # coordinate = data[1] - data[1][data[2]]
