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
├── config.py                   # 参数配置文件
├── main.py                     # 程序主入口
├── preprocess.py               # 数据预处理
├── requirements.txt 
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
```

// TODO

### 可选参数

// TODO
