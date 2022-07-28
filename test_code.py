import numpy as np
import torch
from logzero import logger


def test_pairwise_distance():
    # 欧氏距离
    x1 = torch.Tensor([1, 2, 3])
    x2 = torch.Tensor([1, 2, 4])
    logger.debug(torch.pairwise_distance(x1, x2, eps=0).item())


def test_matrix():
    x1 = torch.Tensor([100., 100., 100., -73., -72.])
    x2 = torch.Tensor([-73., -72.])
    rss_item = x1.tolist()
    length = len(rss_item)
    mx = np.matrix(rss_item * length, dtype=np.float32).reshape(length, length).transpose()
    logger.debug(mx)
    mx = (mx - rss_item) / (np.array(rss_item) + 1)
    result = mx.A.reshape(1, length, length)  # 通道数为 1
    result = torch.from_numpy(result)
    logger.debug(result)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)  # threshold 指定超过多少使用省略号，np.inf代表无限大
    # test_pairwise_distance()
    test_matrix()
