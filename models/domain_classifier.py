import torch.nn as nn
import models as md
from models.grad_reverse import GradReverse


class MD(nn.Module):
    """
    域分类模型MD
    """

    def __init__(self, ap_len, domain_size):
        super(MD, self).__init__()
        self.block1 = md.ConvBlock(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)
        self.max_pool = md.MaxPooling(num_feature=32)
        self.block2 = md.ConvBlock(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.avg_pool = md.AvgPooling(num_feature=64)
        self.bn = nn.BatchNorm2d(32)
        self.soft_max = nn.LogSoftmax(dim=1)
        temp = int(ap_len / 16) * int(ap_len / 16) * 64
        self.fc = nn.Linear(temp, domain_size)

    def forward(self, x, constant):
        batch_size = x.size(0)
        x = GradReverse.grad_reverse(x, constant)
        x = self.bn(x)
        x = self.block1(x)
        x = self.max_pool(x)

        x = self.block2(x)
        x = self.avg_pool(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.soft_max(x)
        return x
