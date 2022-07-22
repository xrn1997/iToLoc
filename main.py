import os
import time

import torch
from logzero import logger

import tools.utils as ut
from tools import utils
from trains import params
from trains.train import Trainer


def main():
    # 训练集
    train_dataset = ut.get_dataset(params.dataset, noise=True)
    # 测试集
    test_dataset = ut.get_dataset(params.dataset, train=False)

    # 初始化模块
    feature_extractor = params.feature_extractor_dict[params.dataset]
    label_predictor = params.label_predictor_dict[params.dataset]
    domain_classifier = params.domain_predictor_dict[params.dataset]

    # 初始化优化器
    optimizer = torch.optim.SGD([{'params': feature_extractor.parameters()},
                                 {'params': label_predictor[0].parameters()},
                                 {'params': label_predictor[1].parameters()},
                                 {'params': label_predictor[2].parameters()},
                                 {'params': domain_classifier.parameters()}],
                                lr=params.learning_rate,
                                momentum=0.9,
                                weight_decay=0.0001)

    # 加载训练参数
    save_path = params.net_save_path
    if not os.path.exists(params.save_dir):
        os.mkdir(params.save_dir)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if os.path.exists(save_path + "/fe.pth"):
        feature_extractor.load_state_dict(torch.load(save_path + "/fe.pth"))
    if os.path.exists(save_path + "/dc.pth"):
        domain_classifier.load_state_dict(torch.load(save_path + "/dc.pth"))
    if os.path.exists(save_path + "/lp1.pth"):
        label_predictor[0].load_state_dict(torch.load(save_path + "/lp1.pth"))
        label_predictor[1].load_state_dict(torch.load(save_path + "/lp2.pth"))
        label_predictor[2].load_state_dict(torch.load(save_path + "/lp3.pth"))

    # 初始化Trainer
    trainer = Trainer(feature_extractor, label_predictor, domain_classifier, optimizer)

    # 日志存储
    utils.log_save(params.save_dir)
    start_time = time.time()
    logger.info("initial_model")
    # 初始化训练
    for epoch in range(params.initial_epochs):
        trainer.train(epoch=epoch, dataset=train_dataset)
        # 验证
        trainer.test(dataset=test_dataset)
        # 保存日志到文件
        utils.log_save(params.save_dir, start_time=start_time, limit=3600)
        start_time = time.time()
    # 用未标记的数据集更新模型
    # trainer.update(initial_dataset=train_dataset, unlabeled_dataset=test_dataset, test_dataset=test_dataset)


if __name__ == '__main__':
    main()
