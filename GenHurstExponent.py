# -*- coding:utf-8 -*-

import numpy as np
import warnings
import pandas as pd
import math
import random
import matplotlib.pyplot as plt


def genhurst(input, q):
    # 防止序列过短导致计算无意义
    L = len(input)
    if L < 100:
        warnings.warn('Data input very short!')

    # 初始化H值和k值
    H = np.zeros((len(range(5, 20)), 1))
    k = 0

    # τmax从5到19天分别计算
    for τmax in range(5, 20):

        # 初始化τmax循环中的x时间序列和mcord序列
        x = np.arange(1, τmax + 1, 1)
        mcord = np.zeros((τmax, 1))

        # 求得mcord序列
        # mcord序列代表q阶矩的增量分布
        for τ in range(1, τmax + 1):
            # Y代表股价序列
            Y = input[np.arange(τ, L + τ, τ) - τ]
            # 序列每隔τ求差值
            Y_diff = input[np.arange(τ, L, τ)] - input[np.arange(τ, L, τ) - τ]
            N = len(Y_diff) + 1
            # X代表时间序列
            X = np.arange(1, N + 1, dtype=np.float64)
            mx = np.sum(X) / N
            SSxx = np.sum(X ** 2) - N * mx ** 2
            my = np.sum(Y) / N
            SSxy = np.sum(np.multiply(X, Y)) - N * mx * my
            # 回归方程中b0（常数项）和b1（斜率）两个系数
            # b1回归系数
            # b0常数项
            b1 = SSxy / SSxx
            b0 = my - b1 * mx
            # 去均值化过程，消除长期趋势对增量间相关性的影响
            ddVd = Y_diff - b1
            VVVd = Y - np.multiply(b1, X) - b0
            mcord[τ - 1] = np.mean(np.abs(ddVd) ** q) / np.mean(np.abs(VVVd) ** q)

        # 双对数回归
        mx = np.mean(np.log10(x))
        # 方差
        SSxx = np.sum(np.log10(x) ** 2) - τmax * mx ** 2
        my = np.mean(np.log10(mcord))
        # 协方差
        SSxy = np.sum(np.multiply(np.log10(x), np.transpose(np.log10(mcord)))) - τmax * mx * my
        # 对数收益率的自相关性
        # 此处自相关性系数即为q*H(q)
        H[k] = SSxy / SSxx
        k = k + 1

    # q*H(q)/q即为H值
    mH = np.mean(H) / q

    return mH


def main():
    # 使用举例
    # Random walk without a drift
    np.random.seed(10)
    n_samples = 999

    series_1 = walk = np.random.standard_normal(size=n_samples)
    series_1[0] = 0.

    for i in range(1, n_samples):
        series_1[i] = series_1[i - 1] + walk[i]

    series_1 = pd.Series(series_1)
    series_1.plot()
    plt.show()
    series_1 = series_1.ravel()
    print("当q为1时，H为：")
    print(genhurst(series_1, 1))
    print("当q为2时，H为：")
    print(genhurst(series_1, 2))

    """
    # Random walk with a drift
    n_samples = 999
    walk = np.random.standard_normal(size=n_samples)
    series_2 = np.empty_like(walk)

    b1 = 1 / 10
    series_2[0] = 0

    for i in range(1, len(walk)):
        series_2[i] = series_2[i - 1] + b1 + walk[i]

    series_2 = pd.Series(series_2)
    series_2.plot()
    plt.show()
    series_2 = series_2.ravel()
    print("当q为1时，H为：")
    print(genhurst(series_2, 1))
    print("当q为2时，H为：")
    print(genhurst(series_2, 2))
    """


if __name__ == "__main__":
    main()
