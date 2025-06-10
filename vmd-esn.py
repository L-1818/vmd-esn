# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:29:07 2022

@author: Acoustics
"""
# from vmdpy import VMD
from VMDnet import VMD
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import time

class ESN():
    def __init__(self, data, N=800, rho=0.7, sparsity=3, T_train=3500, T_predict=300, T_discard=500, eta=1e-6,
                 seed=2020):
        self.data = data
        self.N = N  # reservoir size 库的大小
        self.rho = rho  # spectral radius 谱半径
        self.sparsity = sparsity  # average degree 平均度       sparsity：稀疏性
        self.T_train = T_train  # training steps
        self.T_predict = T_predict  # prediction steps
        self.T_discard = T_discard  # discard first T_discard steps  discard：丢弃
        self.eta = eta  # regularization constant 正则化常数
        self.seed = seed  # random seed

    def initialize(self):
        """
        对连接权矩阵W_IR和W_res进行初始化
        其中W_IR(N*1)是从输入到库的连接权矩阵，W_res(N*N)是从库到输出的连接权矩阵
        """
        if self.seed > 0:
            np.random.seed(self.seed)
        # 生成形状为N * 1的，元素为[-1, 1]之间的随机值的矩阵
        self.W_IR = np.random.rand(self.N, 1) * 2 - 1  # [-1, 1] uniform
        # 生成形状为N * N的，元素为[0, 1]之间的随机值的矩阵
        W_res = np.random.rand(self.N, self.N)
        # 将W_res中大于self.sparsity / self.N的元素置0
        W_res[W_res > self.sparsity / self.N] = 0 \
            # np.linalg.eigvals(W_res)求出W_res的特征值，W_res矩阵除以自身模最大的特征值的模
        W_res /= np.max(np.abs(np.linalg.eigvals(W_res)))
        # 在乘以谱半径
        W_res *= self.rho  # set spectral radius = rho
        self.W_res = W_res

    def train(self):
        u = self.data[:, :self.T_train]  # traning data T_train = 2000
        assert u.shape == (1, self.T_train)
        r = np.zeros((self.N, self.T_train + 1))  # initialize reservoir state r(N*(T_train + 1))
        for t in range(self.T_train):
            # @是Python3.5之后加入的矩阵乘法运算符
            r[:, t + 1] = np.tanh(self.W_res @ r[:, t] + self.W_IR  @ u[:, t])
        # discard first T_discard steps  r丢弃前T_discard步变成r_p
        self.r_p = r[:, self.T_discard + 1:]  # length=T_train-T_discard
        v = self.data[:, self.T_discard + 1:self.T_train + 1]  # target
        self.W_RO = v @ self.r_p.T @ np.linalg.pinv(
            self.r_p @ self.r_p.T + self.eta * np.identity(self.N))
        train_error = np.sum((self.W_RO @ self.r_p - v) ** 2)
        print('Training error: %.4g' % train_error)

    def predict(self):
        u_pred = np.zeros((1, self.T_predict))  # u_pred是形状为(1, self.T_predict)的全零矩阵
        r_pred = np.zeros((self.N, self.T_predict))  # r_pred是形状为(N, self.T_predict)的全零矩阵
        r_pred[:, 0] = self.r_p[:, -1]  # warm start 热启动
        for step in range(self.T_predict - 1):
            u_pred[:, step] = self.W_RO @ r_pred[:, step]
            r_pred[:, step + 1] = np.tanh(self.W_res @
                                          r_pred[:, step] + self.W_IR @ u_pred[:, step])
        u_pred[:, -1] = self.W_RO @ r_pred[:, -1]
        self.pred = u_pred
        return u_pred

    def plot_predict(self):
        ground_truth = self.data[:,
                       self.T_train: self.T_train + self.T_predict]
        plt.figure(figsize=(10, 4))
        plt.plot(self.pred.T, 'b', label='Predict', alpha=0.7)
        plt.plot(ground_truth.T, 'grey', label='True', alpha=0.7)
        plt.rcParams.update({'font.size': 6})
        plt.legend(loc=1)
        plt.show()

    def calc_error(self):
        ground_truth = self.data[:,
                       self.T_train: self.T_train + self.T_predict]
        rmse_list = []
        for step in range(1, self.T_predict + 1):
            error = np.sqrt(
                np.mean((self.pred[:, :step] - ground_truth[:, :step]) ** 2))
            rmse_list.append(error)
        return rmse_list


def moving_average_conv(data, window_size):
   # 卷积平滑滤波
   window = np.ones(window_size) / window_size
   smoothed_data = np.convolve(data, window, mode='same')
   return smoothed_data


if __name__ == "__main__":
    data = np.loadtxt('receiver-1.txt', usecols=1)
    datat = data[3500:3800]
    #data = scipy.signal.savgol_filter(data, 10, 4)
    # data = moving_average_conv(data, 10)
    signal = data[:-300]
    predict_data = data[3500:3800]
    alpha = 2000.0
    tau = 0
    K = 300
    DC = 0
    init = 1
    tol = 1e-7
    st = time.time()
    (u, u_hat, omega) = VMD(signal, alpha, tau, K, DC, init, tol)
    total_pred = np.zeros((1, 300), np.float64)
    for i in range(K):
        train_data = u[i, :]
        train_data = np.reshape(train_data, (1, train_data.shape[0]))
        esn = ESN(train_data)
        esn.initialize()
        esn.train()
        total_pred += esn.predict()
    total_pred = np.reshape(total_pred, [300, ])
    # np.savetxt('vmdesnpre.txt', total_pred)
    # np.savetxt('vmdesntrue.txt', datat)
    ed = time.time()
    print('time:', ed-st)

    fig = plt.figure(1, figsize=(9, 4))

    ax = fig.add_subplot(111)
    ax.xaxis.set_ticks([0, 50, 100, 150, 200, 250, 300])
    ax.set_xticklabels(['3500', '3550', '3600', '3650', '3700', '3750', '3800'])
    font_label = {'family': 'Times New Roman', 'style': 'normal', 'weight': 'normal', 'size': 14}
    font_legend = {'family': 'Times New Roman', 'style': 'normal', 'weight': 'normal', 'size': 14}
    plt.tick_params(labelsize=10)
    plt.plot(datat, 'grey', label='true', alpha=1)
    plt.plot(total_pred, 'b', label='predict', alpha=0.7)
    plt.xlabel('Time Steps', font_label)
    plt.ylabel('Acoustic Pressure/Pa', font_label)
    plt.legend(loc='upper right', prop=font_legend, markerscale=14, framealpha=0)
    err = np.sqrt(np.mean((datat[0:150] - total_pred[0:150]) ** 2))
    err2 = np.sqrt(np.mean((datat[0:50] - total_pred[0:50]) ** 2))
    mape = mean_absolute_percentage_error(datat[0:150], total_pred[0:150])
    r2 = r2_score(datat[0:150], total_pred[0:150])
    mape2 = mean_absolute_percentage_error(datat[0:50], total_pred[0:50])
    r22 = r2_score(datat[0:50], total_pred[0:50])
    print(err, err2, mape, mape2, r2, r22)
    plt.xlim(0, 300)
    plt.show()



