import matplotlib.pyplot as plt
from sklearn import metrics
from config import args_parser
import pandas as pd
import numpy as np
import math

def GetMAPE(y_test, y_hat):
    sum = metrics.mean_absolute_percentage_error(y_test, y_hat)
    return sum
def GetRMSE(y_test, y_hat):
    sum = np.sqrt(metrics.mean_squared_error(y_test, y_hat))
    return sum

def GetMAE(y_test, y_hat):
    sum = metrics.mean_absolute_error(y_test, y_hat)
    return sum

def GetR2(y_test, y_hat):
    sum = metrics.r2_score(y_test, y_hat)
    return sum

if __name__ == '__main__':
    '''
    画图主要是对内容的字体、大小、颜色、清晰度的调整搭配
    图中标题、图例为“小五号黑体” ,也既为9pt
    图中字体为六号宋体，也即为7.5pt
    '''
    plt.rcParams['font.family'] ='SimSun' #设置中文字体为 “宋体”
    plt.rcParams['axes.unicode_minus'] = False  #设置正常显示字符
    font1 = {'family': 'Times New Roman', 'weight': 'normal'}
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # 导入颜色卡
    plt.rcParams['axes.linewidth'] = 0.5  # 设置坐标轴线条粗细

    args = args_parser()
    #导入数据
    y_hat = pd.read_csv("./results/y_hat.csv")
    y_test = pd.read_csv("./results/testY.csv")
    # y_hat = y_hat[:,2]
    # y_test = y_test[:,2]
    print(len(y_hat))

    # -------------------评价指标----------------------
    eva1 = GetRMSE(y_test, y_hat)
    eva2 = GetMAPE(y_test, y_hat)
    eva3 = GetMAE(y_test, y_hat)
    evaluation_metrics = {"RMSE": eva1, "MAPE": eva2, "MAE": eva3}
    print(evaluation_metrics)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(y_hat, label='预测值', color='#308198', linewidth='1.6')
    plt.plot(y_test, label='真实值', color='#FD763F',linewidth='1.6')
    # plt.plot(CLSTM[k:k+48], label='CNN-LSTM', color='#cd5e3c', linewidth='2')
    # plt.plot(RTM_CLSTM[k:k+48], label='RTM-CNN-LSTM', color='#59b9c6')
    plt.legend(loc='upper left', borderaxespad=0.3, labelspacing=0.2, borderpad=0.2, handlelength=1.5, fontsize=14)
    plt.xlabel('时间')
    plt.ylabel('销量')
    # plt.xticks([0, 11, 23, 35, 47], [1, 12, 24, 36, 48])
    plt.tight_layout()
    # plt.show()
    plt.savefig("./results/show.png")

