# -*- coding:utf8 -*-
from sklearn.preprocessing import scale
from scipy.stats import kstest
import numpy as np
import math


class ProcessingMethod:

    def __init__(self):
        pass

    # 标准化：令数据的均值为0，方差为1.输入m*1，返回m*1（默认以列为单位进行处理，但是如果传入的是一行，会处理该行）
    @staticmethod
    def scale(x):
        return scale(x)

    # 中心化：均值为0，协方差阵不变，可用来方便地计算样本的协方差阵
    @staticmethod
    def centralization(x):
        mean = x.mean()
        for i in range(0, x.shape[0]):
            x[i] = x[i]-mean
        return x

    # 极差标准化：均值为0，极差为1
    @staticmethod
    def range_standard(x):
        xmax = x.max()
        xmin = x.min()
        for i in range(0, x.shape[0]):
            if x[i] > 0:
                x[i] = (x[i]-xmin)/(xmax-xmin)
            elif x[i] < 0:
                x[i] = (xmax-x[i])/(xmax-xmin)
        return x

    # 对数变换
    @staticmethod
    def log(x):
        for i in range(0, x.shape[0]):
            x[i] = math.log(x[i])
        return x

    # 离散化：离散为kinds类。离散化的数值为1~kinds。传入m行n列数据，分别进行离散化，并返回m行n列数据
    @staticmethod
    def discretization(y, kinds=10):
        for col in range(0, y.shape[1]):
            ty = sorted(y[:, col])
            divide_std = 1.0 * len(ty) / kinds
            divide_value = []
            for i in range(1, kinds):
                divide_value.append(ty[int(i * divide_std)])
            divide_value.append(np.inf)
            for i in range(0, len(y)):
                for j in range(0, len(divide_value)):
                    if y[i, col] < divide_value[j]:
                        y[i, col] = j + 1
                        break
        return y

    # 正态性检验：返回检验的p值（p值>0.05：样本符合正态分布）
    @staticmethod
    def normal_test(x):
        return kstest(x, cdf='norm')[1]

    # 根据向前查看的天数，获得前查序列
    @staticmethod
    def get_delta(factor, days):
        crn = len(factor)
        for i in range(0, int(days)):  # 获得每次前推的提前量，无法获取的量用0补全
            t_factor = []
            for j in range(0, i + 1):
                t_factor.append(0)
            for j in range(i + 1, crn):
                t_factor.append(factor[j - i - 1][0])
            t_factor = np.transpose([np.array(t_factor)])
            factor = np.hstack((factor, t_factor))
        return factor
