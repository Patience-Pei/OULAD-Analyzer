import argparse
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import *
from preprocess import *

from model import get_model
from evaluation import evaluate_model
def main():
    model_name = Config.model_name   
    print(f"使用模型 {model_name}")
    start_time=time.time()

    print("1.数据加载与预处理")
    raw_df = load_data()
    prep_df = preprocess_data(raw_df)

    print("\n预处理后的数据概览:")
    print(prep_df.head())
    print("\n标签分布:")
    print(prep_df['label'].value_counts())
    print(f"数据总行数: {prep_df.shape[0]}, 总列数: {prep_df.shape[1]}")

    # 定义特征X和标签Y
    X = prep_df.drop(columns=["label"])
    y = prep_df["label"]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TRAIN_TEST_SPLIT, random_state=42, stratify=y)  # 看看这些参数都是啥

    print(f"\n训练集特征维度: {X_train.shape}, 目标维度: {y_train.shape}")
    print(f"测试集特征维度: {X_test.shape}, 目标维度: {y_test.shape}")

    print(f"2.训练模型{model_name}")
    model = get_model(model_name, input_dim=X_train.shape[1] if model_name == 'NeuralNetwork' else None) # ？

    if model_name == 'NeuralNetwork':
        # 神经网络需要 fit 方法的特定参数
        print("开始训练神经网络...")
        # epochs 迭代次数，batch_size 每次更新权重的样本数
        model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=1)
    else:
        # 
        model.fit(X_train, y_train)

    # 模型评估
    print("3.评估模型")
    evaluate_model(model, X_test, y_test, model_name)
    end_time = time.time()
    print(f"\n========== 实验结束: {model_name} (总耗时: {end_time - start_time:.2f} 秒) ==========")


if __name__ == '__main__':
    Config.model_name="DecisionTree"
    main()