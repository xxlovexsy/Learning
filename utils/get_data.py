# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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

    # train = data[:int(len(data) * 0.8)]
    # val = data[int(len(data) * 0.8):int(len(data) * 0.99)]
    # test = data[int(len(data) * 0.8):len(data)]

    train, test = train_test_split(data, test_size=0.5, random_state=42, shuffle=False)
    val = test

    # 归一化
    scaler = MinMaxScaler()
    scaler.fit(data.values)
    joblib.dump(scaler, args.results_path + 'scaler.joblib')
    train = scaler.transform(train.values)
    val = scaler.transform(val.values)
    test = scaler.transform(test.values)

    scaler_y = MinMaxScaler()
    scaler_y.fit(data.loc[:, ['vgnss_x_diff', 'vgnss_y_diff', '纬度_diff', '经度_diff']].values)
    joblib.dump(scaler_y, args.results_path + 'scaler_y.joblib')

    # 生成数据集
    Dtr = reprocess(train, args, step_size=1, shuffle=True)
    Val = reprocess(val, args, step_size=1, shuffle=True)
    Dte = reprocess(test, args, step_size=1, shuffle=False)
    return Dtr, Val, Dte


def get_kfold_data(args):
    print("_"*19 + 'processing' + "_"*19)
    data = load_data()
    args.target_value = "收盘价_元"
    args.target_index = data.columns.get_loc(args.target_value)

    kf = KFold(n_splits=args.kfold, shuffle=False)
    fold_results = []

    for train_index, test_index in kf.split(data):
        # 划分数据集
        train = data.iloc[train_index] if isinstance(data, pd.DataFrame) else data[train_index]
        test = data.iloc[test_index] if isinstance(data, pd.DataFrame) else data[test_index]
        train, val = train_test_split(train, test_size=0.1, random_state=42, shuffle=False)
        # 归一化
        scaler = MinMaxScaler()
        scaler.fit(data[:int(len(data))].values)
        joblib.dump(scaler, args.results_path + 'scaler.joblib')
        train = scaler.transform(train.values)
        val = scaler.transform(val.values)
        test = scaler.transform(test.values)
        # 生成数据集
        Dtr = reprocess(train, args, step_size=1, shuffle=True)
        Val = reprocess(val, args, step_size=1, shuffle=True)
        Dte = reprocess(test, args, step_size=args.output_size, shuffle=False)
        # 将数据集添加到结果列表中
        fold_results.append((Dtr, Val, Dte))

    return fold_results





