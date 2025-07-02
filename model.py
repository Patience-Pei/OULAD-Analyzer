import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from config import *
from preprocess import get_data


def create_logistic_regression():
    # 创建逻辑回归模型
    model = LogisticRegression(
        max_iter=Config.LOGISTIC_REGRESSION_MAX_ITER,
        random_state=Config.RANDOM_STATE
    )

    # 加载数据集并转换成 ndarray 类型
    df = get_data()
    df_array = df.to_numpy()

    # 特征向量与标签分离
    X = df_array[:, :-1]
    y = df_array[:, -1]

    # 分割出训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        test_size=Config.TRAIN_TEST_SPLIT,
        random_state=Config.RANDOM_STATE,
        stratify=y
    )

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练模型
    model.fit(X_train_scaled, y_train)

    # 预测和评估
    y_pred = model.predict(X_test_scaled)
    print(f"分类准确率: {accuracy_score(y_test, y_pred):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 评估每项数据的贡献度
    feature_names = df.columns.tolist()[:-1]
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0],
        'Absolute_Coefficient': abs(model.coef_[0])
    })

    # 按重要性排序
    feature_importance = feature_importance.sort_values('Absolute_Coefficient', ascending=False)

    # 可视化
    plt.figure(figsize=(14, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Absolute_Coefficient'])
    plt.xlabel('Feature Importance (Absolute Coefficient)')
    plt.title('Feature Importance from Logistic Regression')
    plt.show()


class NeuralNetworks(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ## TODO