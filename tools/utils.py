import time

import logzero
import torch
from logzero import logger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import custom_dataset
from trains import params


def get_dataset(dataset, train=True, noise=False):
    """
     获取dataset

    :param noise: 是否添加噪声。如果添加噪声，则每个位置的位置标签数量为3。
    :param dataset: 数据集名称。
    :param train: 是否为训练集。
    :return: dataset
    """
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
        ])
        if train:
            data = custom_dataset.MNISTDataSet(root=params.mnist_path, transform=transform)
        else:
            data = datasets.MNIST(root=params.mnist_path, train=False, transform=transform,
                                  download=True)
    elif dataset == 'UJIndoorLoc':
        data = custom_dataset.UJIndoorLocDataSet(dataset_path=params.UJIndoorLoc_path, train=train, noise=noise)
    else:
        raise Exception('There is no dataset1 named {}'.format(str(dataset)))
    return data


def get_dataloader(dataset, batch_size=params.batch_size, shuffle=True, drop_last=False):
    """
    dataloader的一层封装

    :param dataset: 数据集
    :param batch_size: batch大小
    :param shuffle: 是否乱序
    :param drop_last: 不足batch大小时是否丢弃
    :return: 返回dataloader
    """
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,  # 每次处理的batch大小
                      shuffle=shuffle,  # shuffle的作用是乱序，先顺序读取，再乱序索引。
                      num_workers=1,  # 线程数
                      pin_memory=True,
                      drop_last=drop_last)


def optimizer_scheduler(optimizer, p):
    """
    调整学习率

    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

    return optimizer


def gen_labels(label: int, noise_list: list) -> list:
    """
    根据标签生成n个添加噪声的one-hot标签

    :param noise_list: 噪声向量列表，如果不为空，则用给定的噪声向量生成one-hot标签
    :param label: 原始标签，int类型
    :param types: 分类数量
    :return: 添加噪声后的标签列表,该列表中有3个one-hot标签向量（列表）
    """
    temp = []
    for k in range(len(noise_list)):
        noise = noise_list[k]
        noise[label] = noise[label] + 1
        noise = torch.Tensor(noise)
        noise = noise / noise.sum()
        temp.append(noise.tolist())
    return temp


def gen_noise(types=10, num=3) -> list:
    """
    生成n个添加噪声的one-hot标签

    :param num: 生成的标签数量
    :param types: 分类数量
    :return: 添加噪声后的标签列表,该列表中有3个one-hot标签向量（列表）
    """
    temp = []
    for k in range(num):
        noise = torch.relu(torch.randn(types) * params.std)
        temp.append(noise.tolist())
    return temp


def log_save(save_dir, start_time=time.time(), limit=0):
    """
    日志存储

    :param save_dir: 存储目录
    :param start_time: 开始时间
    :param limit: 隔多久新开一个文件，单位为秒
    """
    if time.time() - start_time > limit:
        logzero.logfile(save_dir + "/output_" + str(time.time()) + ".log")
        start_time = time.time()


# 测试用代码
if __name__ == '__main__':
    t = 3
    test_noise = gen_noise(t, 20)
    for i in test_noise:
        logger.debug(i)
    result = gen_labels(1, noise_list=test_noise)
    for i in result:
        logger.debug(i)
