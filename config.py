import torch

class Config:
    RANDOM_STATE = 42                   # 随机数种子

    # 超参数配置
    TRAIN_TEST_SPLIT = 0.2              # 测试数据的比例
    LOGISTIC_REGRESSION_MAX_ITER = 1000 # 逻辑回归迭代轮数          
    EPOCHS = 30                         # 神经网络训练的 epoch 数
    BATCH_SIZE = 512                    # 训练批次大小
    LEARNING_RATE = 0.001               # 学习率
    WEIGHT_DECAY = 1e-5
    DROPOUT_RATE = 0.2                  # dropout 层中的丢弃概率
    MIXED_PRECISION = True              # 启用混合精度训练
    N_WEEKS = 4                         # 选用前 N 周的数据
    VERBOSE = False                     # 是否显示 MLP 训练日志（仅对 sklearn 模型）
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 运行设备

    # 项目路径参数
    DATA_DIR = 'anonymisedData'         # 数据集路径
    RESULT_DIR = 'results'              # 结果输出路径
