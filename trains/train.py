import math

import numpy as np
import torch.cuda
from torch import nn
from torch.autograd import Variable
from logzero import logger
import models as md
import params
import custom_dataset
from tools import utils


class Trainer:
    def __init__(self, feature_extractor, label_predictor: [], domain_classifier, optimizer):
        # GPU or CPU
        if torch.cuda.is_available() and params.use_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # 加载模型到指定设备
        self.fe = feature_extractor.to(self.device)
        self.dc = domain_classifier.to(self.device)
        self.lp = [label_predictor[0].to(self.device), label_predictor[1].to(self.device),
                   label_predictor[2].to(self.device)]

        # 损失函数
        self.class_criterion = md.OneHotNLLLoss(reduction='multiply')
        self.domain_criterion = md.OneHotNLLLoss()
        self.update_criterion = nn.NLLLoss()
        self.tri_net_criterion = md.TriNetLoss()
        # 优化器
        self.optimizer = optimizer

    def train(self, epoch, dataset, mv=-1) -> None:
        """
        训练数据集

        :param epoch: 训练轮数
        :param dataset: 数据集
        :param mv: 默认值为-1，即正常训练且保存训练模型，mv不为-1时只训练mv，mv为0时训练mv和ms。这里
            mv为标签预测器，ms为特征提取器。
        """

        # 设置模式
        self.fe.train()
        self.dc.train()
        self.lp[0].train()
        self.lp[1].train()
        self.lp[2].train()
        # dataloader
        dataloader = utils.get_dataloader(dataset=dataset, drop_last=True)
        # steps
        start_steps = epoch * len(dataloader)
        total_steps = params.initial_epochs * len(dataloader)
        # 损失
        output_loss = 0
        for batch_idx, data in enumerate(dataloader):

            # 用于调整学习率
            p = float(batch_idx + start_steps) / total_steps
            constant = 2. / (1. + np.exp(-params.gamma * p)) - 1
            # 优化器
            self.optimizer = utils.optimizer_scheduler(self.optimizer, p)
            self.optimizer.zero_grad()
            inputs, positions, position_labels, domain_labels = data
            # 输入
            inputs = Variable(inputs).to(self.device, non_blocking=True)

            if mv == 0:
                # 提取特征
                feature = self.fe(inputs).data
            else:
                feature = self.fe(inputs)
            if mv == -1:
                # 位置预测
                position_preds1 = self.lp[0](feature)
                position_preds2 = self.lp[1](feature)
                position_preds3 = self.lp[2](feature)
                # 位置标签
                position_label1 = Variable(position_labels[:, 0]).to(self.device, non_blocking=True)
                position_label2 = Variable(position_labels[:, 1]).to(self.device, non_blocking=True)
                position_label3 = Variable(position_labels[:, 2]).to(self.device, non_blocking=True)
                # 位置分类器损失
                position_loss1 = self.class_criterion(position_preds1, position_label1)
                position_loss2 = self.class_criterion(position_preds2, position_label2)
                position_loss3 = self.class_criterion(position_preds3, position_label3)
                # 位置损失
                position_loss = self.tri_net_criterion(position_loss1, position_loss2, position_loss3)
                # 预测域
                domain_preds = self.dc(feature, constant)
                # 域标签
                domain_labels = Variable(domain_labels).to(self.device, non_blocking=True)
                # 域分类(器)损失
                domain_loss = self.domain_criterion(domain_preds, domain_labels)
                # 总损失
                loss = position_loss + params.theta * domain_loss
                # 反向传播
                loss.backward()
            else:
                # 预测标签
                position_preds = self.lp[mv](feature)
                # 标签
                position_labels = Variable(inputs).to(self.device, non_blocking=True)
                # 损失
                loss = self.update_criterion(position_preds, position_labels)
                # 反向传播
                loss.backward()
            # 更新参数
            self.optimizer.step()
            # 保存当前损失，用于打印输出
            output_loss = loss.item()

        logger.info('epoch:{}\tLoss: {:.6f}\t'.format(epoch, output_loss))
        if mv == -1:
            save_path = params.tri_net_save_path
            torch.save(self.fe.state_dict(), save_path + "/fe.pth")
            torch.save(self.dc.state_dict(), save_path + "/dc.pth")
            torch.save(self.lp[0].state_dict(), save_path + "/lp1.pth")
            torch.save(self.lp[1].state_dict(), save_path + "/lp2.pth")
            torch.save(self.lp[2].state_dict(), save_path + "/lp3.pth")

    def test(self, dataset) -> None:
        """
         测试数据集

        :param dataset: 数据集
        """
        logger.debug("test")
        # 设置模式
        self.fe.eval()
        self.dc.eval()
        self.lp[0].eval()
        self.lp[1].eval()
        self.lp[2].eval()

        # 设置变量
        correct1 = 0.0
        correct2 = 0.0
        correct3 = 0.0

        # dataloader
        dataloader = utils.get_dataloader(dataset=dataset)

        for batch_idx, data in enumerate(dataloader):
            inputs, positions, position_labels, domain_labels = data
            inputs = Variable(inputs).to(self.device, non_blocking=True)
            # 输入
            inputs = Variable(inputs).to(self.device, non_blocking=True)
            # 提取特征
            feature = self.fe(inputs)
            # 位置预测
            position_pred1 = self.lp[0](feature).data.max(1, keepdim=True)[1]
            position_pred2 = self.lp[1](feature).data.max(1, keepdim=True)[1]
            position_pred3 = self.lp[2](feature).data.max(1, keepdim=True)[1]
            # 位置标签
            position_label1 = Variable(position_labels[:, 0]).to(self.device, non_blocking=True)
            position_label2 = Variable(position_labels[:, 1]).to(self.device, non_blocking=True)
            position_label3 = Variable(position_labels[:, 2]).to(self.device, non_blocking=True)
            # 域预测
            domain_preds = self.dc(feature).data.max(1, keepdim=True)[1]
            # 域标签
            domain_labels = Variable(domain_labels).to(self.device, non_blocking=True)
            # 结果
            correct1 += position_pred1.eq(label.data.view_as( position_pred1)).cpu().sum()
            correct2 +=  position_pred2.eq(label.data.view_as( position_pred1)).cpu().sum()
            correct3 +=  position_pred3.eq(label.data.view_as( position_pred1)).cpu().sum()

        logger.debug('\n预测器1的正确率: {}/{} ({:.4f}%)'.format(
            correct1, len(dataloader.dataset), 100. * float(correct1) / len(dataloader.dataset)
        ))
        logger.debug('\n预测器2的正确率: {}/{} ({:.4f}%)'.format(
            correct2, len(dataloader.dataset), 100. * float(correct2) / len(dataloader.dataset)
        ))
        logger.debug('\n预测器3的正确率: {}/{} ({:.4f}%)'.format(
            correct3, len(dataloader.dataset), 100. * float(correct3) / len(dataloader.dataset)
        ))

    def update(self, initial_dataset, unlabeled_dataset, test_dataset) -> None:
        """
        更新模型。

        :param test_dataset: 测试用数据集
        :param initial_dataset:  初始数据集。
        :param unlabeled_dataset:   未标记的数据集。
        """
        logger.debug("update")
        # setup models
        self.fe.train()
        self.dc.train()
        self.lp[0].train()
        self.lp[1].train()
        self.lp[2].train()
        flag = 1
        sigma = params.sigma_0
        mu = unlabeled_dataset
        lv = [[], [], [], []]
        for i in initial_dataset:
            lv[0].append([i[0], i[1][0]])
            lv[1].append([i[0], i[1][1]])
            lv[2].append([i[0], i[1][2]])
        for j in lv[0]:
            lv[3].append([j[0], j[1].argmax()])
        for t in range(1, params.T + 1):
            # n_t = min(1000 * pow(2, t), params.U)
            n_t = params.U
            if n_t == params.U:
                if t % 4 == 0:
                    for epoch in range(params.initial_epochs):
                        self.train(epoch=epoch, dataset=initial_dataset)
                    flag = 1
                    sigma = sigma - 0.05
                    continue
            if flag == 1:
                flag = 0
                sigma_t = sigma - params.sigma_os
            else:
                sigma_t = sigma
            for v in range(0, 3):
                logger.info("第" + str(t) + "轮，未标记的数据集大小: " + str(n_t) + "，训练模型" + str(v))
                plv = self.labeling((v + 1) % 3, (v + 2) % 3, mu, n_t, sigma_t)
                logger.debug("labeling plv: " + str(len(plv)))
                plv = self.des(plv, (v + 1) % 3, (v + 2) % 3)
                logger.debug("des plv: " + str(len(plv)))
                data_list = lv[3] + plv
                dataset = custom_dataset.List2DataSet(data_list)
                for epoch in range(params.update_epochs):
                    self.train(epoch=epoch, dataset=dataset, mv=v)
                self.test(test_dataset)

    def labeling(self, mj, mh, mu, nt, sigma_t):
        """
        预测标签。两个预测器结果一致，并且平均最大后验概率大于σ_t.

        :param mj:  三个预测model之一。
        :param mh:  三个预测model之一，且与mj不同。
        :param mu: 未标记的数据集
        :param nt:  数据集大小（从mu中拿nt大小的未标记数据）。
        :param sigma_t: 过滤不确定的伪标签的阈值参数。
        :return: 打好伪标签的数据集。
        """
        logger.debug("labeling")
        dataloader = utils.get_dataloader(dataset=mu, shuffle=False)

        plv = []

        for batch_idx, data in enumerate(dataloader):
            if batch_idx * params.batch_size > nt:
                break
            inputs, _ = data
            inputs = Variable(inputs).to(self.device, non_blocking=True)

            feature = self.fe(inputs)

            preds_j = self.lp[mj](feature).data.max(1, keepdim=True)
            preds_h = self.lp[mh](feature).data.max(1, keepdim=True)
            equals = preds_j[1].eq(preds_h[1])
            confident = ((preds_j[0] + preds_h[0]) / 2 >= math.log(sigma_t))
            result = equals * confident
            for idx, i in enumerate(result):
                if i.item():
                    plv.append([inputs[idx].cpu(), preds_j[1][idx].cpu().squeeze()])
        return plv

    def des(self, plv, mj, mh):
        """
        判断标签是否稳定。

        :param plv: 待处理的有伪标签的数据集
        :param mj:  三个预测model之一。
        :param mh:  三个预测model之一，且与mj不同。
        :return: 稳定的伪标签数据集。
        """
        logger.debug("des")
        if len(plv) == 0:
            return []
        # 设置模式
        self.fe.train()
        self.lp[0].train()
        self.lp[1].train()
        self.lp[2].train()
        dataset = custom_dataset.List2DataSet(plv)
        dataloader = utils.get_dataloader(dataset, drop_last=True)
        new_plv = []
        for index, data in enumerate(dataloader):
            inputs, labels = data
            inputs = Variable(inputs).to(self.device, non_blocking=True)
            labels = Variable(labels).to(self.device, non_blocking=True)
            # 记录预测错误的次数
            k = torch.zeros(inputs.shape[0])
            for time in range(0, 9):
                feature = self.fe(inputs)
                preds_j = self.lp[mj](feature).data.max(1, keepdim=True)[1]
                preds_h = self.lp[mh](feature).data.max(1, keepdim=True)[1]
                error_j = preds_j.ne(labels.data.view_as(preds_j)).cpu().squeeze()
                error_h = preds_h.ne(labels.data.view_as(preds_h)).cpu().squeeze()
                k = k + error_j + error_h
            for n in range(0, inputs.shape[0]):
                if k[n] <= 3:
                    new_plv.append([inputs[n].cpu(), labels[n].cpu()])
        return new_plv
