from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import tensorflow as tf
import pandas as pd
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    评估模型并打印分类报告、混淆矩阵和ROC AUC分数。
    Args:
        model: 训练好的模型实例。
        X_test (pd.DataFrame): 测试集特征。
        y_test (pd.Series): 测试集真实标签。
        model_name (str): 模型的名称，用于打印报告。
    """
    print(f"\n--- {model_name} 模型评估 ---")

    # 对于Keras模型，需要特殊处理预测
    if isinstance(model, tf.keras.Model):
        y_prob = model.predict(X_test).flatten()
        y_pred = (y_prob > 0.5).astype(int) # 转换为二元预测
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] # 获取正类的概率

    print(f"\n{model_name} 分类报告:")
    #print(classification_report(y_test, y_pred, target_names=["非学业困难", "学业困难"]))
    #print(f"{model_name} 混淆矩阵:\n", confusion_matrix(y_test, y_pred))

    report_dict = classification_report(y_test, y_pred, target_names=["非学业困难", "学业困难"], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose() # 转置后更符合表格习惯
    print(report_df.to_string()) # 使用 to_string() 避免截断

    print(f"\n{model_name} 混淆矩阵:\n", confusion_matrix(y_test, y_pred))
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"{model_name} ROC AUC 分数: {roc_auc:.4f}")
    except ValueError:
        print("无法计算 ROC AUC 分数 (可能因为测试集中只有一个类别)。")

    # 如果是基于树的模型，打印特征重要性
    if hasattr(model, 'feature_importances_'):
        print(f"\n--- {model_name} 特征重要性 (Top 10) ---")
        feature_importances = model.feature_importances_
        features = X_test.columns
        
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        print(importance_df.head(10))