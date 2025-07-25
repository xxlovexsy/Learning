# -*- coding:utf-8 -*-
"""
   Author:          yeong
   Email:           yeong66@126.com
   Date:            2024-12-16
   Version:         1.0.0
   Copyright:       (c) 2024 XYZ Corporation. All rights reserved.
   License:         MIT License
"""


from config import args_parser

import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from utils.util import test

import torch
import joblib
import os

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)




class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(np.array(data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(labels), dtype=torch.float32)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

def reprocess(dataset, args, step_size, shuffle):
    seq_len, batch_size, output_size= args.seq_len, args.batch_size, args.output_size
    seqs, labels = [], []
    for i in range(0, len(dataset) - seq_len - output_size + 1, step_size):
        train_seq = dataset[i:i+seq_len, :]  # dataX是一个三维矩阵（timeseries, timestep, features）
        seqs.append(train_seq)
        train_label = dataset[i+seq_len: i+seq_len+1, [6,7,9,10]]  # dataY同样是一个三维矩阵(timeseries, features)
        labels.append(train_label)
    data_set = SimpleDataset(seqs, labels)
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)
    return data_loader

def select_features(data):
    print(data.columns.tolist())
    # 检查行中缺失值超过30%的行，并删除这些行
    data.dropna(thresh=len(data.columns) * 0.7, inplace=True)

    data = data.loc[:,['vgnss_x', 'vgnss_y', 'vgnss_z', '纬度_x', '经度_x', '高度_x',
                       'vgnss_x_diff', 'vgnss_y_diff', 'vgnss_z_diff', '纬度_diff', '经度_diff', '高度_diff',
                       'w_x', 'w_y', 'w_z', 'f_x', 'f_y', 'f_z', 'Pitch', 'Roll', 'Yaw', 'V_E', 'V_N', 'V_U',]]
                       # '纬度_y', '经度_y', '高度_y', '陀螺漂移误差_x', '陀螺漂移误差_y', '陀螺漂移误差_z']]
    # 缺失值三项插值填充
    # data = data.interpolate(method='linear')
    # 缺失值填充为0
    # data = data.fillna(0)

    # column_list = list(data.columns)
    # fill_ = KNNImputer(n_neighbors=5, weights='distance')  # KNN按距离填充
    # data = fill_.fit_transform(data)
    # data = pd.DataFrame(columns=column_list, data=data)
    #data.rename(columns={'体感温度': 'tgtemp'}, inplace=True)

    return data

def load_data():
    data = pd.read_excel('./data/merged_df.xlsx')
    data = select_features(data)
    print(data.head(5))
    return data


def get_data(args):
    print("_"*19 + 'processing' + "_"*19)
    data = load_data()
    print(data.shape)

    test_data = data

    # 归一化
    scaler = joblib.load(args.results_path + 'scaler.joblib')
    scaler.fit(test_data.values)
    test_data = scaler.transform(test_data.values)

    scaler_y = MinMaxScaler()
    scaler_y.fit(data.loc[:, ['vgnss_x_diff', 'vgnss_y_diff', '纬度_diff', '经度_diff']].values)
    joblib.dump(scaler_y, args.results_path + 'scaler_y.joblib')

    # 生成数据集
    Dte = reprocess(test_data, args, step_size=1, shuffle=False)
    return Dte


def main():
    args = args_parser()
    print("model args :", args)
    Dte = get_data(args)

    test(Dte, args)



if __name__ == '__main__':
    main()

