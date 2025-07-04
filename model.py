import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import time
from config import *
from preprocess import get_data, model_data
from evaluate import shap_evaluate

def create_logistic_regression():
    '''使用逻辑回归模型进行预测'''
    # 创建逻辑回归模型
    model = LogisticRegression(
        max_iter=Config.LOGISTIC_REGRESSION_MAX_ITER,
        random_state=Config.RANDOM_STATE
    )

    # 获取可处理数据
    df, X_train_scaled, y_train, X_test_scaled, y_test = model_data()

    # 训练模型
    model.fit(X_train_scaled, y_train)

    # 预测和评估
    y_pred = model.predict(X_test_scaled)
    print(f"分类准确率: {accuracy_score(y_test, y_pred):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 最终使用 SHAP 值评估模型
    shap_evaluate(model, 'LogisticRegression')

    # if hasattr(model, "predict_proba"):
    #     y_prob = model.predict_proba(X_test_scaled)[:, 1]
    #     roc_auc = roc_auc_score(y_test, y_prob)
    #     print(f"ROC AUC分数: {roc_auc:.4f}")
    # else:
    #     print("模型不支持predict_proba，无法计算ROC AUC。")

    # # 评估每项数据的贡献度
    # feature_names = df.columns.tolist()[:-1]
    # feature_importance = pd.DataFrame({
    #     'Feature': feature_names,
    #     'Coefficient': model.coef_[0],
    #     'Absolute_Coefficient': abs(model.coef_[0])
    # })

    # # 按重要性排序
    # feature_importance = feature_importance.sort_values('Absolute_Coefficient', ascending=False)

    # # 可视化
    # plt.figure(figsize=(14, 6))
    # plt.barh(feature_importance['Feature'], feature_importance['Absolute_Coefficient'])
    # plt.xlabel('Feature Importance (Absolute Coefficient)')
    # plt.title('Feature Importance from Logistic Regression')
    # plt.gca().invert_yaxis()
    # plt.show()


class CustomFCNN(nn.Module):
    '''自定义的全连接神经网络模型'''
    def __init__(self, input_size=23, dropout_rate=Config.DROPOUT_RATE):
        super(CustomFCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 第一层
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        # 第二层
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        # 第三层
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        # 输出层
        x = self.fc4(x)
        return x
    
def get_dataset():
    # 获取可处理数据
    df, X_train_scaled, y_train, X_test_scaled, y_test = model_data()

    # 转换为 tensor
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = Config.BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

def train_model(model):
    '''对神经网络模型进行训练的函数'''
    # 定义损失函数、优化器和学习率调度器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=Config.EPOCHS)

    # 混合精度训练
    if Config.MIXED_PRECISION and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    # 获取数据加载器
    train_loader, test_loader = get_dataset()

    model.train()
    model.to(Config.DEVICE)
    max_acc = 0.0
    start_time = time.time()
    for epoch in range(Config.EPOCHS):
        epoch_start_time = time.time()

        # 训练一个 epoch
        train_loss, train_acc = train_epoch(model, criterion, optimizer, scheduler, scaler, train_loader)
        # 测试模型
        test_loss, test_acc = test_model(model, scaler, criterion, test_loader)
        if test_acc > max_acc:
            max_acc = test_acc

        # 更新学习率
        scheduler.step()

        epoch_time = time.time() - epoch_start_time

        # 打印进度
        print(f'Epoch {epoch+1:3d}/{Config.EPOCHS} | '
                f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
                f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | '
                f'LR: {optimizer.param_groups[0]["lr"]:.6f} | '
                f'Time: {epoch_time:.2f}s')
        
    total_time = time.time() - start_time
    print(f'\nTraining complete. Elapsed time: {total_time:.2f}s. Highest test accuracy: {max_acc:.4f}')

    # 最终使用 SHAP 值评估模型
    shap_evaluate(model, 'NeuralNetworks')


def train_epoch(model, criterion, optimizer, scheduler, scaler, train_loader):
    '''训练一个 epoch'''
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for data, target in train_loader:
        data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
        optimizer.zero_grad()

        if scaler is not None:
            # 混合精度训练
            with torch.amp.autocast('cuda'):
                output = model(data)
                loss = criterion(output.squeeze(), target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 常规训练
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * data.size(0)
        probs = torch.sigmoid(output).flatten().cpu().detach().numpy()
        preds = (probs > 0.5).astype(int)
        all_labels.extend(target.cpu().detach().numpy())
        all_preds.extend(preds)

    epoch_loss = running_loss / len(train_loader)
    correct = np.sum(np.array(all_labels) == np.array(all_preds))
    epoch_acc = correct / len(all_labels)

    return epoch_loss, epoch_acc

def test_model(model, scaler, criterion, test_loader):
    '''测试模型'''
    model.eval()
    test_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    output = model(data)
                    loss = criterion(output.squeeze(), target)
            else:
                output = model(data)
                loss = criterion(output.squeeze(), target)

            test_loss += loss.item() * data.size(0)
            probs = torch.sigmoid(output).flatten().cpu().detach().numpy()
            preds = (probs > 0.5).astype(int)
            all_labels.extend(target.cpu().detach().numpy())
            all_preds.extend(preds)

    test_loss /= len(test_loader)
    correct = np.sum(np.array(all_labels) == np.array(all_preds))
    test_acc = correct / len(all_labels)

    return test_loss, test_acc

def create_neural_networks():
    # 设置随机种子
    torch.manual_seed(Config.RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.RANDOM_STATE)

    # 创建模型并进行训练
    model = CustomFCNN()
    train_model(model)


def create_randomforest():
    """
    随机森林
    """
    # 创建模型
    model = RandomForestClassifier(n_estimators=100, 
                                   random_state=Config.RANDOM_STATE, 
                                   class_weight='balanced')
    # 训练并评估模型
    # evaluate(model)
    df, X_train_scaled, y_train, X_test_scaled, y_test = model_data()
    model.fit(X_train_scaled, y_train)

    # 预测和评估
    y_pred = model.predict(X_test_scaled)
    print(f"分类准确率: {accuracy_score(y_test, y_pred):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 最终使用 SHAP 值评估模型
    shap_evaluate(model, 'RandomForest')


def create_gradient_boosting():
    """
    梯度提升树
    """
    model = GradientBoostingClassifier(random_state=Config.RANDOM_STATE)
    # evaluate(model)
    df, X_train_scaled, y_train, X_test_scaled, y_test = model_data()
    model.fit(X_train_scaled, y_train)

    # 预测和评估
    y_pred = model.predict(X_test_scaled)
    print(f"分类准确率: {accuracy_score(y_test, y_pred):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 最终使用 SHAP 值评估模型
    shap_evaluate(model, 'gradient_boosting')