# -*- coding:utf-8 -*-

import copy
import os

import json, joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from tqdm import tqdm
from itertools import chain
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)


def plt_loss(train_losses, val_losses, epochs):
    # 绘制训练和验证损失
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, color='b', label='Train Loss')  # 假设train_loss是存储损失的列表
    plt.plot(range(1, epochs + 1), val_losses, color='r', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./results/loss.png')
    plt.close()
    # plt.show()

def train(Dtr, Val, model, optimizer, criterion, args):
    print("_"*20 + 'training' + "_"*20)
    epochs = args.epochs
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    all_preds, all_targets = [], []
    train_losses, val_losses = [], []
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        for inputs, targets in Dtr:
            inputs, targets = inputs.to(args.device), targets.squeeze(1).to(args.device)
            optimizer.zero_grad()
            y_pred = model(inputs)
            loss = criterion(y_pred, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(Dtr)
        train_losses.append(train_loss)  # 记录训练损失

        # 验证模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (inputs, targets) in Val:
                inputs, targets = inputs.to(args.device), targets.squeeze(1).to(args.device)

                y_pred = model(inputs)
                loss = criterion(y_pred, targets)
                val_loss += loss.item()

                r2_v = r2_score(targets.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
                print(f'epoch [{epoch + 1}/{epochs}], r2_v: {r2_v:.4f}')
                if epoch == epochs - 1:
                    all_preds.extend(y_pred.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
            val_loss = val_loss / len(Val)
            val_losses.append(val_loss)  # 记录验证损失

        scheduler.step()
        print(f'epoch [{epoch + 1}/{epochs}], train_loss: {train_loss:.4f}, val_Loss: {val_loss:.4f}')
    plt_loss(train_losses, val_losses, epochs)
    torch.save(model, args.results_path +"best_model.pth")
    # state = {'models': best_model.state_dict(), 'optimizer': optimizer.state_dict()}


def plt_comparison(y, y_hat, args):
    # 遍历每个特征
    for i in range(y_hat.shape[1]):
        plt.figure(figsize=(10, 8))  # 创建一个新的图形
        plt.plot(y[:, i], label='true', color='r', linestyle='-', linewidth=1, marker='*', markersize=5) # 绘制真实值
        plt.plot(y_hat[:, i], label='pred', color='b', linestyle='-', linewidth=1, marker='+', markersize=5)  # 绘制预测值
        plt.grid(axis='y')
        plt.title(f'feature_{i + 1}_comparison')  # 设置标题
        plt.xlabel('x')  # X坐标标签
        plt.ylabel('y')  # Y坐标标签
        plt.legend()  # 显示图例
        plt.savefig(os.path.join(args.results_path, f'feature_{i + 1}_comparison.png'))
        # plt.show()  # 显示图形
        plt.close()  # 关闭图形

def save_evaluation_results(y, y_hat, args):
    # 检查y_hat是否为None，以确定是否成功进行了预测
    if y_hat is not None:
        eva1 = get_rmse(y, y_hat)
        eva3 = get_mape(y, y_hat)
        eva2 = get_mae(y, y_hat)
        eva4 = get_r2(y, y_hat)
        evaluation_metrics = {
            "RMSE": eva1,
            "MAPE": eva2,
            "MAE": eva3,
            "R2": eva4,
        }
        print("反归一化前指标：", evaluation_metrics)
        # ----------------------------------------------
        scaler_y = joblib.load(args.results_path + 'scaler_y.joblib')
        # 反归一化
        y, y_hat = scaler_y.inverse_transform(y), scaler_y.inverse_transform(y_hat)
        # 反归一化
        # m, n = scaler.data_max_[args.target_index], scaler.data_min_[args.target_index]
        # y, y_hat = (m - n) * y + n, (m - n) * y_hat + n
        # 反标准化
        # m, n = scaler.scale_[args.target_index], scaler.mean_[args.target_index]
        # y, y_hat = y * m + n, y_hat * m + n
        # ----------------------------------------------
        y = pd.DataFrame(y)
        y.to_csv('./results/y_test.csv', index=False)
        y_hat = pd.DataFrame(y_hat)
        y_hat.to_csv('./results/y_hat.csv', index=False)
        # ----------------------------------------------
        eva1 = get_rmse(y, y_hat)
        eva3 = get_mape(y, y_hat)
        eva2 = get_mae(y, y_hat)
        eva4 = get_r2(y, y_hat)
        evaluation_metrics = {
            "RMSE": eva1,
            "MAPE": eva2,
            "MAE": eva3,
            "R2": eva4,
        }
        print(evaluation_metrics)
        with open('./results/evaluation_metrics.json', 'w') as file:
            json.dump(evaluation_metrics, file)


def test(Dte, args):
    print("_"*20 + 'testing' + "_"*20)
    # model = TransformerModel(args).to(args.device)
    # model.load_state_dict(torch.load(args.path)['models'])

    def load_premodel(model_path):
        model = None
        if os.path.isfile(model_path):
            try:
                model = torch.load(model_path, map_location = args.device)
                print("模型成功导入")
                # model.summary()
            except Exception as e:
                print(f"加载模型时出错：{e}")
                return None  # 如果模型加载出错，返回None
        if model is not None:
            return model  # 返回预测结果
        else:
            print("模型导入失败")
            pass

    # 加载预测
    model_path = args.results_path + "best_model.pth"
    model = load_premodel(model_path).to(args.device)
    # 测试模型
    model.eval()
    pred, y = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(Dte, 0):
            inputs = inputs.to(args.device)
            targets = targets.squeeze(1).to(args.device)
            y_pred = model(inputs)

            y_pred = y_pred.cpu().tolist()
            pred.extend(y_pred)
            targets = targets.cpu().tolist()
            y.extend(targets)

    y, y_hat = np.array(y), np.array(pred)
    print("y:",y.shape,"y_hat:",y_hat.shape)

    # 保存预测结果
    save_evaluation_results(y, y_hat, args)
    plt_comparison(y, y_hat, args)


def get_rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))

def get_mape(y, pred):
    return mean_absolute_percentage_error(y, pred)

def get_mae(y, pred):
    return mean_absolute_error(y, pred)

def get_mse(y, pred):
    return mean_squared_error(y, pred)

def get_r2(y, pred):
    return r2_score(y, pred)
