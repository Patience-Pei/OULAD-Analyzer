import matplotlib.pyplot as plt
import pandas as pd
from config import *
from preprocess import get_data

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def model_data():
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
    return df,X_train_scaled, y_train, X_test_scaled, y_test

def evaluate(model):
    df,X_train_scaled, y_train, X_test_scaled, y_test = model_data()
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
        'Importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)


    # 按重要性排序
    # feature_importance = feature_importance.sort_values('Absolute_Coefficient', ascending=False)

    # 可视化
    plt.figure(figsize=(14, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])

    plt.xlabel('Feature Importance (Absolute Coefficient)')
    plt.title('Feature Importance from Random Forest')
    plt.show()
def create_randomforest():
    # 创建模型
    model = RandomForestClassifier(n_estimators=100, 
                                   random_state=Config.RANDOM_STATE, 
                                   class_weight='balanced')
    
    # 训练并评估模型
    evaluate(model)