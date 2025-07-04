import os
import shap
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from config import *
from preprocess import model_data

def evaluate(model, model_name, test_data, labels):
    '''
    评估模型
    '''
    # 使用模型进行预测并计算准确率、ROC AUC 分数
    predictions = model.predict(test_data)
    acc = accuracy_score(labels, predictions)
    auc_score = roc_auc_score(labels, predictions)
    report = classification_report(labels, predictions)

    print(f"分类准确率: {acc:.4f}")
    print(f"ROC AUC 分数: {auc_score:.4f}")
    print("\n分类报告:")
    print(report)

    # 确保结果目录存在
    save_path = os.path.join(Config.RESULT_DIR, model_name)
    os.makedirs(save_path, exist_ok=True)
    report_path = os.path.join(save_path, 'classification_report.txt')

    # 保存分类报告到文本文件
    with open(report_path, 'w') as f:
        f.write(f"Prediction Accuracy: {acc:.4f}\n")
        f.write(f"ROC AUC Score: {auc_score:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)

    # 最终使用 SHAP 值评估模型
    shap_evaluate(model, model_name)

def shap_evaluate(model, model_name):
    '''
    使用 SHAP 值对模型进行评估和可视化
    '''
    df, X_train_scaled, y_train, X_test_scaled, y_test = model_data()
    # 确保特征名称是字符串列表
    feature_names = df.columns.tolist()[:-1]
    feature_names = [str(name) for name in feature_names]
    # 确保结果目录存在
    save_path = os.path.join(Config.RESULT_DIR, model_name)
    os.makedirs(save_path, exist_ok=True)
    
    # 获取完整测试集
    test_data, test_labels = torch.Tensor(X_test_scaled), torch.Tensor(y_test)
    test_data = test_data[:100].to(Config.DEVICE)  # 限制样本数以提高性能
    
    # 准备背景数据 (用于解释)
    background_data = torch.Tensor(X_train_scaled[:500]).to(Config.DEVICE)
    
    # 创建预测函数和解释器
    # if model_name == 'NeuralNetworks':
    #     def model_predict(x):
    #         model.eval()
    #         with torch.no_grad():
    #             x_tensor = torch.tensor(x, dtype=torch.float32).to(Config.DEVICE)
    #             logits = model(x_tensor)
    #             probabilities = torch.sigmoid(logits).cpu().numpy()
    #             preds = (probabilities > 0.5).astype(int)
    #         return preds

    #     explainer = shap.KernelExplainer(
    #         model_predict, 
    #         shap.sample(background_data.cpu().numpy(), 100)
    #     )
    # else:
    #     explainer = shap.KernelExplainer(
    #         model.predict,
    #         shap.sample(background_data.cpu().numpy(), 100)
    #     )
    explainer = shap.KernelExplainer(
        model.predict,
        shap.sample(background_data.cpu().numpy(), 100)
    )
    
    # 计算SHAP值
    print("Calculating SHAP values...")
    test_samples = test_data.cpu().numpy()
    shap_values = explainer.shap_values(
        test_samples, 
        nsamples=100
    )
    
    # 对于二分类，SHAP值可能以列表形式返回
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # 取第一个输出
    
    # 可视化分析 - 多种方法展示特征贡献
    print("Creating visualizations...")
    
    # 1. 特征重要性摘要图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        test_samples, 
        feature_names=feature_names,
        plot_type="dot",
        show=False
    )
    plt.title("Feature Importance Summary")
    plt.tight_layout()
    summary_path = os.path.join(save_path, "shap_summary.png")
    plt.savefig(summary_path)
    plt.close()
    print(f"Saved summary plot: {summary_path}")
    
    # 2. 蜂群图 (Beeswarm plot) - 展示特征值分布和影响方向
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        test_samples, 
        feature_names=feature_names,
        plot_type="violin",
        show=False
    )
    plt.title("Feature Impact Distribution")
    plt.tight_layout()
    beeswarm_path = os.path.join(save_path, "shap_beeswarm.png")
    plt.savefig(beeswarm_path)
    plt.close()
    print(f"Saved beeswarm plot: {beeswarm_path}")
    
    # 3. 特征重要性条形图
    # 计算平均绝对SHAP值
    if len(shap_values) > 1:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).flatten()
    
    # 确保mean_abs_shap是一维数组
    if len(mean_abs_shap.shape) > 1:
        mean_abs_shap = mean_abs_shap.flatten()
    
    # 获取排序后的索引
    sorted_idx = np.argsort(mean_abs_shap)  # 从小到大排序
    
    # 创建排序后的特征名称和值列表
    sorted_features = []
    sorted_values = []
    
    # 逐个处理每个索引
    for idx in sorted_idx:
        # 确保索引是整数
        if isinstance(idx, np.ndarray):
            idx = idx.item()
        sorted_features.append(feature_names[idx])
        sorted_values.append(mean_abs_shap[idx])
    
    # 创建水平条形图
    plt.figure(figsize=(12, 8))
    plt.barh(
        sorted_features, 
        sorted_values,
        color='skyblue'
    )
    plt.xlabel("Mean Absolute SHAP Value")
    plt.title("Feature Importance Ranking")
    plt.tight_layout()
    bar_path = os.path.join(save_path, "shap_feature_importance.png")
    plt.savefig(bar_path)
    plt.close()
    print(f"Saved feature importance bar plot: {bar_path}")
    
    # 保存SHAP值结果
    shap_values_path = os.path.join(save_path, "shap_values.npy")
    np.save(shap_values_path, shap_values)
    
    # 创建综合报告
    create_shap_report(
        shap_values, 
        test_samples, 
        test_labels[:100].cpu().numpy(),
        feature_names,
        explainer.expected_value,
        save_path
    )
    
    print("SHAP evaluation completed successfully!")

def create_shap_report(shap_values, features, labels, feature_names, base_value, save_path):
    '''创建包含关键洞察的SHAP报告'''
    # 确保特征名称是字符串
    feature_names = [str(name) for name in feature_names]
    
    report = {
        "overall_insights": [],
        "feature_analysis": [],
        "sample_analysis": []
    }
    
    # 1. 计算特征重要性
    if len(shap_values) > 1:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).flatten()
    
    # 确保mean_abs_shap是一维数组
    if len(mean_abs_shap.shape) > 1:
        mean_abs_shap = mean_abs_shap.flatten()
    
    # 获取最重要的前5个特征
    if len(mean_abs_shap) > 0:
        top_features_idx = np.argsort(mean_abs_shap)[::-1][:min(5, len(mean_abs_shap))]
        report["overall_insights"].append(
            f"Top {len(top_features_idx)} most important features: {', '.join(feature_names[i] for i in top_features_idx)}"
        )
    else:
        top_features_idx = []
        report["overall_insights"].append("No features available for analysis")
    
    # 2. 特征级分析
    for i in top_features_idx:
        # 确保索引是整数
        if isinstance(i, np.ndarray):
            i = i.item()
        
        feature = feature_names[i]
        
        # 提取特征值和对应的SHAP值
        feature_values = features[:, i]
        shap_values_for_feature = shap_values[:, i]
        
        # 确保数组形状一致
        if feature_values.ndim > 1:
            feature_values = feature_values.flatten()
        if shap_values_for_feature.ndim > 1:
            shap_values_for_feature = shap_values_for_feature.flatten()
        
        # 确保数组长度一致
        min_length = min(len(feature_values), len(shap_values_for_feature))
        feature_values = feature_values[:min_length]
        shap_values_for_feature = shap_values_for_feature[:min_length]
        
        # 计算特征值与SHAP值的相关性
        if min_length > 1:
            # 使用协方差矩阵计算相关性
            cov_matrix = np.cov(feature_values, shap_values_for_feature)
            if cov_matrix.size >= 4:  # 确保有足够的元素
                var1 = cov_matrix[0, 0]
                var2 = cov_matrix[1, 1]
                cov = cov_matrix[0, 1]
                
                if var1 > 0 and var2 > 0:
                    corr = cov / np.sqrt(var1 * var2)
                else:
                    corr = 0
            else:
                corr = 0
        else:
            corr = 0
        
        # 分析特征方向性
        pos_mask = shap_values_for_feature > 0
        neg_mask = shap_values_for_feature < 0
        
        pos_impact = np.mean(shap_values_for_feature[pos_mask]) if np.any(pos_mask) else 0
        neg_impact = np.mean(shap_values_for_feature[neg_mask]) if np.any(neg_mask) else 0
        
        analysis = {
            "feature": feature,
            "mean_abs_impact": float(mean_abs_shap[i]),
            "directionality_correlation": float(corr),
            "positive_impact_percent": float(pos_impact * 100),
            "negative_impact_percent": float(neg_impact * 100),
            "interpretation": ""
        }
        
        # 添加解释性描述
        if corr > 0.3:
            analysis["interpretation"] = (
                f"Higher values of {feature} generally increase the prediction."
            )
        elif corr < -0.3:
            analysis["interpretation"] = (
                f"Higher values of {feature} generally decrease the prediction."
            )
        else:
            analysis["interpretation"] = (
                f"The relationship between {feature} and prediction is complex and non-linear."
            )
        
        report["feature_analysis"].append(analysis)
    
    # 3. 样本级分析
    # 获取预测概率
    if len(shap_values) > 0:
        predictions = base_value + np.sum(shap_values, axis=1)
        pred_labels = (predictions > 0.5).astype(int)
    else:
        predictions = np.array([])
        pred_labels = np.array([])
    
    # 找到最具代表性的样本
    if len(shap_values) > 0:
        impact_scores = np.sum(np.abs(shap_values), axis=1)
        max_impact_idx = np.argmax(impact_scores)
        min_impact_idx = np.argmin(impact_scores)
    else:
        max_impact_idx = 0
        min_impact_idx = 0
    
    # 正确和错误分类的样本
    correct_pred_idx = []
    incorrect_pred_idx = []
    
    for i in range(len(labels)):
        if i < len(pred_labels) and pred_labels[i] == labels[i]:
            correct_pred_idx.append(i)
        elif i < len(pred_labels):
            incorrect_pred_idx.append(i)
    
    # 添加样本分析
    sample_analysis = [
        {
            "type": "most_impact",
            "sample_idx": int(max_impact_idx),
            "actual_label": int(labels[max_impact_idx]),
            "predicted_label": int(pred_labels[max_impact_idx]) if max_impact_idx < len(pred_labels) else -1,
            "prediction_prob": float(predictions[max_impact_idx]) if max_impact_idx < len(predictions) else -1,
            "top_features": []
        },
        {
            "type": "least_impact",
            "sample_idx": int(min_impact_idx),
            "actual_label": int(labels[min_impact_idx]),
            "predicted_label": int(pred_labels[min_impact_idx]) if min_impact_idx < len(pred_labels) else -1,
            "prediction_prob": float(predictions[min_impact_idx]) if min_impact_idx < len(predictions) else -1,
            "top_features": []
        }
    ]
    
    # 添加正确/错误分类的样本
    if correct_pred_idx:
        sample_analysis.append({
            "type": "correctly_classified",
            "sample_idx": int(correct_pred_idx[0]),
            "actual_label": int(labels[correct_pred_idx[0]]),
            "predicted_label": int(pred_labels[correct_pred_idx[0]]) if correct_pred_idx[0] < len(pred_labels) else -1,
            "prediction_prob": float(predictions[correct_pred_idx[0]]) if correct_pred_idx[0] < len(predictions) else -1,
            "top_features": []
        })
    
    if incorrect_pred_idx:
        sample_analysis.append({
            "type": "incorrectly_classified",
            "sample_idx": int(incorrect_pred_idx[0]),
            "actual_label": int(labels[incorrect_pred_idx[0]]),
            "predicted_label": int(pred_labels[incorrect_pred_idx[0]]) if incorrect_pred_idx[0] < len(pred_labels) else -1,
            "prediction_prob": float(predictions[incorrect_pred_idx[0]]) if incorrect_pred_idx[0] < len(predictions) else -1,
            "top_features": []
        })
    
    # 为每个样本添加最重要的特征
    for sample in sample_analysis:
        idx = sample["sample_idx"]
        if idx < len(shap_values) and idx < features.shape[0]:
            sample_shap = shap_values[idx]
            
            # 获取对该样本影响最大的特征
            abs_shap = np.abs(sample_shap)
            top_feature_idx = np.argsort(abs_shap)[::-1][:min(3, len(abs_shap))]
            
            for i in top_feature_idx:
                # 确保索引是整数
                if isinstance(i, np.ndarray):
                    i = i.item()
                
                sample["top_features"].append({
                    "feature": feature_names[i],
                    "value": float(features[idx, i]),
                    "shap_value": float(sample_shap[i]),
                    "impact": "positive" if sample_shap[i] > 0 else "negative"
                })
    
    report["sample_analysis"] = sample_analysis
    
    # 4. 保存报告
    report_path = os.path.join(save_path, "shap_analysis_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Created comprehensive SHAP report: {report_path}")
    
    return report