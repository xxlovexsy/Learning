# -*- coding:utf-8 -*-
"""
   Author:          yeong
   Email:           yeong66@126.com
   Date:            2024-12-16
   Version:         1.0.0
   Copyright:       (c) 2024 XYZ Corporation. All rights reserved.
   License:         MIT License
"""

from utils.get_data import get_data, get_kfold_data  #获取数据
from config import args_parser #用于解析命令行参数获配置文件中超参数
from utils.util import train, test  #导入训练与测试函数
from models.transformer_lstm import TransformerLSTM
import torch.nn as nn
import numpy as np
import torch

seed = 1   #固定随机种子，确保结果可复现
np.random.seed(seed)
torch.manual_seed(seed)


#程序入口，设置参数，加载数据，建立模型，调用训练与测试流程
def main():
    args = args_parser()
    print("model args :", args)
    Dtr, Val, Dte = get_data(args)
    model = TransformerLSTM(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss().to(args.device)

    #train(Dtr, Val, model, optimizer, loss_fn, args)
    test(Dte, args)



if __name__ == '__main__':
    main()
