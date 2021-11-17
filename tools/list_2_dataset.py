import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from logzero import logger
import custom_dataset
from trains import params


class List2DataSet(Dataset):
    """
    给数据集列表套一层壳
    """

    def __init__(self, list_data):
        self.data = list_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]


# 测试
if __name__ == '__main__':
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
    ])
    plv = [[torch.Tensor([1, 1]), torch.Tensor([1])], [torch.Tensor([2, 2]), torch.Tensor([1])],
           [torch.Tensor([3, 3]), torch.Tensor([3])], [torch.Tensor([4, 4]), torch.Tensor([4])],
           [torch.Tensor([5, 5]), torch.Tensor([5])], [torch.Tensor([6, 6]), torch.Tensor([6])],
           [torch.Tensor([7, 7]), torch.Tensor([7])], [torch.Tensor([8, 8]), torch.Tensor([8])]]
    dataset1 = custom_dataset.List2DataSet(plv)
    dataloader = DataLoader(dataset=dataset1,
                            batch_size=2,  # 每次处理的batch大小
                            shuffle=False,  # shuffle的作用是乱序，先顺序读取，再乱序索引。
                            num_workers=1,  # 线程数
                            pin_memory=True)
    for loader in dataloader:
        data, label = loader
        logger.info(data)
        logger.info(label)

