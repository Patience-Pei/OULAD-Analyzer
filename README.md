## 项目结构
```
labFinal/
├── anonymisedData/             # 数据集
│   ├── assessments.csv
│   ├── courses.csv
│   ├── studentAssessment.csv
│   ├── studentInfo.csv
│   ├── studentRegistration.csv
│   ├── studentVle.csv
│   └── vle.csv
├── results/                    # 按模型类别分类存放评估结果的目录
│   ├── GradientBoosting/
│   ├── LogisticRegression/
│   ├── NeuralNetworks/
│   └── RandomForest/
├── config.py                   # 参数配置文件
├── main.py                     # 程序统一主入口
├── preprocess.py               # 数据预处理
├── model.py                    # 算法模型构建
├── evaluate.py                 # 算法模型评估
├── requirements.txt            # 程序环境依赖
└── README.md
```

## 运行步骤
1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行程序：
```bash
python main.py --model LR   # 运行逻辑回归模型
python main.py --model NN   # 运行神经网络模型
python main.py --model RF   # 运行随机森林模型
python main.py --model GB   # 运行梯度提升树模型
```

## 运行结果说明

- 运行成功后程序会输出模型的分类准确率、ROC AUC 分数以及分类报告
- 模型 SHAP 评估的结果会写入到 `results` 目录下
- 在所有步骤完成后，程序输出模型的训练时间、评估时间以及总运行时间