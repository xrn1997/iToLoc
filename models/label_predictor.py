import torch.nn as nn
import models as md


class M1(nn.Module):
    """
    标签预测模型M1
    """

    def __init__(self, ap_len, position_size):
        super(M1, self).__init__()
        self.block1 = md.ConvBlock(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)
        self.max_pool = md.MaxPooling(num_feature=32)
        self.block2 = md.ConvBlock(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.avg_pool = md.AvgPooling(num_feature=64)
        self.bn = nn.BatchNorm2d(32)
        self.soft_max = nn.LogSoftmax(dim=1)
        temp = int(ap_len/16) * int(ap_len/16) * 64
        self.fc = nn.Linear(temp, position_size)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.bn(x)
        x = self.block1(x)
        x = self.max_pool(x)

        x = self.block2(x)
        x = self.avg_pool(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.soft_max(x)
        return x


class M2(nn.Module):
    """
    标签预测模型M2
    """

    def __init__(self, ap_len, position_size):
        super(M2, self).__init__()
        self.block1 = md.ConvBlock(in_channels=32, out_channels=32, kernel_size=(5, 5), padding=2)
        self.max_pool = md.MaxPooling(num_feature=32)
        self.block2 = md.ConvBlock(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=2)
        self.avg_pool = md.AvgPooling(num_feature=64)
        self.bn = nn.BatchNorm2d(32)
        self.soft_max = nn.LogSoftmax(dim=1)
        temp = int(ap_len/16) * int(ap_len/16) * 64
        self.fc = nn.Linear(temp, position_size)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.bn(x)
        x = self.block1(x)
        x = self.max_pool(x)

        x = self.block2(x)
        x = self.avg_pool(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.soft_max(x)
        return x


class M3(nn.Module):
    """
    标签预测模型M3
    """

    def __init__(self, ap_len, position_size):
        super(M3, self).__init__()
        self.block1 = md.ResidualBlock(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)
        self.max_pool = md.MaxPooling(num_feature=32)
        self.block2 = md.ResidualBlock(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.avg_pool = md.AvgPooling(num_feature=64)
        self.bn = nn.BatchNorm2d(32)
        self.soft_max = nn.LogSoftmax(dim=1)
        temp = int(ap_len / 16) * int(ap_len / 16) * 64
        self.fc = nn.Linear(temp, position_size)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.bn(x)
        x = self.block1(x)
        x = self.max_pool(x)

        x = self.block2(x)
        x = self.avg_pool(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.soft_max(x)
        return x
