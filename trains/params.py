import models as md

# 训练参数
use_gpu = True  # 是否使用GPU
dataset_mean = [0.5]  # 均值
dataset_std = [0.5]  # 标准差
std = 0.001  # 输出涂抹标准差
batch_size = 6  # batch块大小
initial_epochs = 50  # 训初始练轮数
update_epochs = 30  # 模型每次更新训练轮数
learning_rate = 0.0001  # 学习率
gamma = 10  # γ
theta = 0.3  # θ
beta = 0.5  # β
T = 30  # 模型更新轮数
sigma_0 = 0.999  # σ_0
sigma_os = 0.01  # σ_os
initial_size = 10000  # 初始数据集大小
U = 9937  # 未标记数据大小
# 路径参数
data_root = './custom_dataset'
mnist_path = data_root + '/MNIST'
UJIndoorLoc_path = data_root + '/UJIndoorLoc/'
save_dir = './experiment'
net_save_path = save_dir + "/iToLoc"
# 数据集
dataset = "UJIndoorLoc"
# 数据集参数
ap_len = 520
co_size = 933
domain_size = 19
# 模型
feature_extractor_dict = {'UJIndoorLoc': md.ME()}
label_predictor_dict = {'UJIndoorLoc': [md.M1(co_size), md.M2(co_size), md.M3(co_size)]}
domain_predictor_dict = {'UJIndoorLoc': md.MD(domain_size)}

