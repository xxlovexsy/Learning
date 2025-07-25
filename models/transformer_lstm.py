# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #运行设备GPU有限
seed = 1
torch.manual_seed(seed)


#定义位置编码  为输入序列中每个时间步添加一个唯一的向量编码，帮助Transformer感知位置信息
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].squeeze(0)
        return x


class TransformerLSTM(nn.Module):
    def __init__(self, args, num_encoder_layers=6,num_decoder_layers=6,dropout=0.1, max_len=5000):
        super(TransformerLSTM, self).__init__()
        # embedding
        self.embedding = nn.Linear(args.input_size, args.d_model)
        # Transformer
        self.pos_encoder = PositionalEncoding(args.d_model, max_len=max_len)
        self.transformer = nn.Transformer(
            d_model=args.d_model,
            nhead=2,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=4 * args.input_size,
            batch_first=True,
            dropout=dropout,
            device=device,
            )
        self.trans_lstm = nn.LSTM(args.d_model, args.d_model, num_layers=1, batch_first=True, bidirectional=False)
        self.output_fc = nn.Sequential(
            nn.Linear(args.d_model, args.d_model),   #维度不变的线性映射
            nn.ReLU(),  #激活函数
            nn.Dropout(p=0.1),    #Dropout正则化
            nn.Linear(args.d_model, args.output_size)
            )
        self.src_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    ##前向传播函数
    def forward(self, src):

        trans_features = self.embedding(src)
        trans_features = self.pos_encoder(trans_features)

        # Transformer 编码器-解码器部分
        x = self.transformer.encoder(trans_features)
        # x = x.flatten(start_dim=1)
        x, _ = self.trans_lstm(x)
        x = x[:, -1, :]
        output = self.output_fc(x)

        return output
