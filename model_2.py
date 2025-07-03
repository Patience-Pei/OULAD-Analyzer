import matplotlib.pyplot as plt
import pandas as pd
from config import *
from preprocess import get_data, model_data
from evaluate import shap_evaluate

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def evaluate(model):
    df,X_train_scaled, y_train, X_test_scaled, y_test = model_data()
    model.fit(X_train_scaled, y_train)
    # 预测和评估
    y_pred = model.predict(X_test_scaled)
    print(f"分类准确率: {accuracy_score(y_test, y_pred):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC分数: {roc_auc:.4f}")
    else:
        print("模型不支持predict_proba，无法计算ROC AUC。")

    # 评估每项数据的贡献度
    feature_names = df.columns.tolist()[:-1]
    if hasattr(model, "feature_importances_"):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        plt.figure(figsize=(14, 6))
        plt.barh(feature_importance['Feature'], feature_importance['Importance'])
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance from {model.__class__.__name__}')
        plt.gca().invert_yaxis()
        plt.show()
    else:
        print(f"模型 {model.__class__.__name__} 不支持特征重要性显示。")

def create_randomforest():
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
    model = GradientBoostingClassifier(random_state=Config.RANDOM_STATE)
    evaluate(model)


# 对随机森林调参
def tune_random_forest():
    df, X_train, y_train, X_test, y_test = model_data()

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced']
    }

    model = RandomForestClassifier(random_state=Config.RANDOM_STATE)
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        scoring='accuracy',   # 或 
                        cv=5,
                        n_jobs=-1,
                        verbose=2)
    
    grid.fit(X_train, y_train)
    print("最优参数：", grid.best_params_)
    print("最佳 accuracy 分数：", grid.best_score_)

    print("\n在测试集上的表现：")
    best_model = grid.best_estimator_
    evaluate(best_model)
def tune_gradient_boosting():
    df, X_train, y_train, X_test, y_test = model_data()

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.05],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3],
        'subsample': [0.8, 1.0],
        'max_features': ['sqrt', 'log2']
    }

    model = GradientBoostingClassifier(random_state=Config.RANDOM_STATE)

    grid = GridSearchCV(model,
                        param_grid,
                        scoring='accuracy', # 
                        cv=5,
                        n_jobs=-1,
                        verbose=2)

    grid.fit(X_train, y_train)

    print("最优参数：", grid.best_params_)
    print("最佳 accuracy 分数：", grid.best_score_)

    print("\n在测试集上的表现：")
    best_model = grid.best_estimator_
    evaluate(best_model)

# tune_gradient_boosting()