import torch
import torch.nn as nn


class OneHotNLLLoss(nn.Module):
    """
    one-hot的交叉熵损失函数
    """

    def __init__(self, reduction='mean'):
        super(OneHotNLLLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        # y是标签，x是预测值
        if self.reduction == 'sum':
            n = 1
        elif self.reduction == 'mean':
            n = len(y)
        elif self.reduction == 'multiply':
            # 什么也不做，只保留最原始的-y log(x)
            return -x * y
        else:
            raise Exception('reduction 参数错误')
        return -torch.sum(x * y) / n


class TriNetLoss(nn.Module):
    """
    tri_net交叉熵损失函数
    """

    def __init__(self):
        super(TriNetLoss, self).__init__()

    @staticmethod
    def forward(x, y, z):
        return torch.sum(x + y + z) / len(x)