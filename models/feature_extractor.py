import torch.nn as nn
import models as md


class ME(nn.Module):
    """
    特征提取模型ME
    """
    def __init__(self):
        super(ME, self).__init__()
        self.conv_block = md.ConvBlock(kernel_size=(3, 3), out_channels=32, padding=1, in_channels=1)
        self.max_pool = md.MaxPooling(num_feature=32)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.max_pool(x)
        return x
