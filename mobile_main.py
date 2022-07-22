import time

import torch
from logzero import logger
from torch.autograd import Variable

import models
import models as md
import tools.utils as ut
from tools import utils
from trains import params


class Mobile_Trainer:
    def __init__(self):
        # GPU or CPU
        if torch.cuda.is_available() and params.use_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # 初始化模块
        self.net = models.mobilenet_v2(num_classes=933).to(self.device)
        # 初始化优化器
        self.optimizer = torch.optim.SGD([{'params': self.net.parameters()}],
                                         lr=params.learning_rate,
                                         momentum=0.9,
                                         weight_decay=0.0001)
        # 损失函数
        self.criterion = md.OneHotNLLLoss(smx=True)

    def train(self, epoch, dataset) -> None:

        self.net.train()
        # dataloader
        dataloader = utils.get_dataloader(dataset=dataset, drop_last=True)
        for batch_idx, data in enumerate(dataloader):
            p = float(batch_idx) / len(dataloader)
            # 优化器
            self.optimizer = utils.optimizer_scheduler(self.optimizer, p)
            self.optimizer.zero_grad()
            inputs, positions, position_labels, domain_labels = data
            # 输入
            inputs = Variable(inputs).to(self.device, non_blocking=True)
            position_label = Variable(position_labels[:, 0]).to(self.device, non_blocking=True)
            # 提取特征
            preds = self.net(inputs)
            loss = self.criterion(preds, position_label)
            # 反向传播
            loss.backward()
            # 更新参数
            self.optimizer.step()
            # print loss
            if (batch_idx + 1) % 10 == 0:
                logger.info('epoch:{}\t[{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch,
                    batch_idx * len(inputs),
                    len(dataloader.dataset),
                    100. * batch_idx / len(dataloader),
                    loss.item(),
                ))

    def test(self, dataset) -> None:
        self.net.eval()
        correct = 0.0
        # dataloader
        dataloader = utils.get_dataloader(dataset=dataset)
        for batch_idx, data in enumerate(dataloader):
            inputs, positions, position_labels, domain_labels = data
            # 输入
            inputs = Variable(inputs).to(self.device, non_blocking=True)
            # 提取特征
            preds = self.net(inputs).data.max(1, keepdim=True)[1]
            # 位置标签
            position_labels = Variable(position_labels).to(self.device, non_blocking=True)
            # 结果
            correct += preds.eq(position_labels.data.view_as(preds)).cpu().sum()
        logger.debug('\n预测器的正确率: {}/{} ({:.4f}%)'.format(
            correct, len(dataloader.dataset), 100. * float(correct) / len(dataloader.dataset)
        ))


if __name__ == '__main__':
    # 训练集
    train_dataset = ut.get_dataset(params.dataset, noise=True)

    # 验证集
    test_dataset = ut.get_dataset(params.dataset, train=False)
    test2_dataset = ut.get_dataset(params.dataset, train=True, noise=False)
    # 初始化Trainer
    trainer = Mobile_Trainer()
    # 日志存储
    utils.log_save(params.save_dir)
    start_time = time.time()
    logger.info("initial_model")
    # 初始化训练
    for ep in range(params.initial_epochs):
        # 训练
        trainer.train(epoch=ep, dataset=train_dataset)
        # 测试
        trainer.test(dataset=test_dataset)
        trainer.test(dataset=test2_dataset)
        # 保存日志到文件
        utils.log_save(params.save_dir, start_time=start_time, limit=3600)
        start_time = time.time()
