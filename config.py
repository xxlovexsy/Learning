# -*- coding:utf-8 -*-
import argparse
import torch
import os

##定义训练或测试会用到的参数  参数配置函数
def args_parser():

    def path_type(dir_path):
        if not os.path.exists(dir_path):
            raise argparse.ArgumentTypeError("The directory {} does not exist!".format(dir_path))
        return dir_path

    ##创建ArgumentParser对象，用于解析命令行参数
    parser = argparse.ArgumentParser()

    ##训练流程超参数
    parser.add_argument('--results_path', type=path_type, default='./results/', help='Path to save the results')
    parser.add_argument('--kfold', type=int, default=10, help='kfold')
    parser.add_argument('--epochs', type=int, default=80, help='epochs')
    parser.add_argument('--seq_len', type=int, default=6, help='seq len')
    ##模型结构参数
    parser.add_argument('--input_size', type=int, default=24, help='input dimension')
    parser.add_argument('--d_model', type=int, default=32, help='d_model')  #Transformer与LSTM隐藏单元数
    parser.add_argument('--output_size', type=int, default=4, help='output dimension')
    ##优化器与训练策略配置
    parser.add_argument('--lr', type=float, default=0.0004, help='learning rate')  #学习率
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')  #批量大小
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')#优化器类型
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')#权重衰减系数（L2正则），用于防止过拟合
    parser.add_argument('--step_size', type=int, default=50, help='step size') #学习滤调度的步长
    parser.add_argument('--gamma', type=float, default=0.25, help='gamma')  #学习率调度的衰减因子

    args = parser.parse_args()

    return args
