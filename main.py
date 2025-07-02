import argparse

from config import *
from model import *
from model_2 import *


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()

    # 模型参数
    parser.add_argument('-m', '--model', type=str, default='LR',
                        choices=['LR', 'LogisticRegression', 'NN', 'NeuralNetworks','RF','RandomForest'],
                        help='选择模型，默认为逻辑回归模型')
    

    return parser.parse_args()

def main():
    '''程序统一主入口'''
    args = parse_arguments()

    if args.model == 'LR' or args.model == 'LogisticRegression':
        create_logistic_regression()
    elif args.model == 'NN' or args.model == 'NeuralNetworks':
        pass
    elif args.model == 'RF'or args.model == 'RandomForest':
        create_randomforest()



if __name__ == '__main__':
    main()