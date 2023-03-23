import math as m  # 数学
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import time
from pyqpanda import *
import tensorflow as tf
from tensorflow.keras.utils import normalize
from sklearn.decomposition import PCA
import cv2 as cv

start = time.time()

def numerical_gradient(f, params, x, n):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(params)  # 生成和参数形状相同的数组
    for idx in range(params.size):
        tmp_val = params[idx]
        # f(x+h)的计算
        params[idx] = tmp_val + h
        fxh1, ypre = f(params, x, n)
        # f(x-h)的计算
        params[idx] = tmp_val - h
        fxh2, ypre = f(params, x, n)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        params[idx] = tmp_val  # 还原值
    return grad


def MAE(Y, t):
    return np.sum(np.absolute(t - Y))


def MSE(Y, t):
    return 0.5 * (np.sum(Y - t) ** 2)


def RMSE(Y, t):
    return np.sqrt(np.sum(Y - t) ** 2)


def cross_entropy_error(y, t):
    delta = 1e-7  ##添加一个微小值可以防止负无限大(np.log(0))的发生
    return -np.sum(t * np.log(y + delta))  # + np.sum((1-t) * np.log(1-y))  #t是标签，y是网络的输出


# 数据输入U_in矩阵
def U_in(qubits, X_t):
    # print(len(X_t))
    for i in range(len(X_t)):
        if X_t[i] > 1:
            X_t[i] = 1
        elif X_t[i] < -1:
            X_t[i] = -1

    circuit = create_empty_circuit()
    theta_in = np.zeros(len(X_t))
    for i in range(len(X_t)):
        theta_in[i] = m.acos(X_t[i])
    for i in range(3):
        circuit << RY(qubits[i], theta_in[0]) \
        << RY(qubits[i], theta_in[1]) \
        << RY(qubits[i], theta_in[2]) \
        << RY(qubits[i], theta_in[3]) \
        << RY(qubits[i], theta_in[4]) \
        << RY(qubits[i], theta_in[5])
    return circuit


# 参数矩阵，有3*6=18个参数
def U_theta(qubits, params):
    circuit = create_empty_circuit()
    for i in range(6):
        circuit << RX(qubits[i], params[3 * i]) \
        << RZ(qubits[i], params[3 * i + 1]) \
        << RX(qubits[i], params[3 * i + 2])
    return circuit


# 哈密顿量模拟第一部分，有6个参数
def H_X(qubits, params):
    circuit = create_empty_circuit()
    for i in range(6):
        circuit << RX(qubits[i], params[i])
    return circuit


# 哈密顿量模拟第二部分，有6个参数
def H_ZZ(qubits, params):
    circuit = create_empty_circuit()
    for i in range(5):
        circuit << CNOT(qubits[i], qubits[i + 1]) \
        << RZ(qubits[i + 1], params[i]) \
        << CNOT(qubits[i], qubits[i + 1])
    circuit << CNOT(qubits[5], qubits[0]) \
    << RZ(qubits[0], params[5]) \
    << CNOT(qubits[5], qubits[0])
    return circuit


# 整个参数线路，共18+6+6=30个参数
def QRNN_VQC(qubits, params):
    params1 = params[0: 18]
    params2 = params[18: 18 + 6]
    params3 = params[18 + 6: 30]
    circuit = create_empty_circuit()
    for i in range(3):
        circuit << U_theta(qubits, params1) \
        << H_X(qubits, params2) \
        << H_ZZ(qubits, params3)
    return circuit


# 损失函数,共30+3个参数，其中前30个为量子线路参数，最后3个为经典参数
def loss(params, X_t, n):
    LOSS = 0
    zhenfu = np.array([1, 0, 0, 0, 0, 0, 0, 0])

    for i in range(n):
        qvm = CPUQVM()  # 建立一个局部的量子虚拟机
        qvm.init_qvm()  # 初始化量子虚拟机
        qubits = qvm.qAlloc_many(6)
        prog = QProg()
        circuit = create_empty_circuit()

        # circuit << U_in(qubits, X_t[i])  # 数据输入
        circuit << amplitude_encode([qubits[0], qubits[1], qubits[2]], X_t[i])
        circuit << amplitude_encode([qubits[3], qubits[4], qubits[5]], zhenfu)  # 后三个比特的编码
        circuit << QRNN_VQC(qubits, params[0: 30])
        # circuit << QRNN_VQC(qubits, params[30*i: 30*i +30])

        prog << circuit

        qubit0_prob = qvm.prob_run_list(prog, qubits[0], -1)
        qubit1_prob = qvm.prob_run_list(prog, qubits[1], -1)
        qubit2_prob = qvm.prob_run_list(prog, qubits[2], -1)

        # 坍缩到1的概率直接当均值
        qubit0_avrage = qubit0_prob[1]

        # 这里只用第一个比特的概率
        Y_prediction = qubit0_avrage

        # 求后三个比特最后的状态振幅
        zhenfu_2 = qvm.prob_run_list(prog, [qubits[3], qubits[4], qubits[5]], -1)
        zhenfu = np.sqrt(np.array(zhenfu_2))

        # 释放局部虚拟机
        qvm.finalize()

    # LOSS = m.fabs(Y_prediction - X_t[n] )/ X_t[n]
    LOSS = RMSE(Y_prediction, Y_train[i])
    # LOSS = cross_entropy_error(Y_prediction, Y_train[i])

    return LOSS, Y_prediction


def Accuarcy(params, n):
    # test_iterations = len(Y_test_in)
    print("lr = ", lr)
    test_iterations = 200
    print("test iterations = ", test_iterations)

    #  这里两个用于记录测试集的真实值和预测值
    Y_pre_history_test = []
    Y_true_history_test = []

    Ei_2_sum = 0  # 误差平方和初始化
    a = 0
    for j in range(test_iterations):
        zhenfu = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        X_test_n = normalize(X_test_in[j], axis=1)
        X_t = np.array(X_test_n)

        for i in range(n):
            qvm = CPUQVM()  # 建立一个局部的量子虚拟机
            qvm.init_qvm()  # 初始化量子虚拟机
            qubits = qvm.qAlloc_many(6)
            # cbits = qvm.cAlloc_many(6)
            prog = QProg()
            circuit = create_empty_circuit()

            # circuit << U_in(qubits, X_t[i])  # 数据输入
            circuit << amplitude_encode([qubits[0], qubits[1], qubits[2]], X_t[i])
            circuit << amplitude_encode([qubits[3], qubits[4], qubits[5]], zhenfu)  # 后三个比特的编码
            # circuit << QRNN_VQC(qubits, params[30*i: 30*i +30])
            circuit << QRNN_VQC(qubits, params[0: 30])
            prog << circuit

            qubit0_prob = qvm.prob_run_list(prog, qubits[0], -1)
            qubit1_prob = qvm.prob_run_list(prog, qubits[1], -1)
            qubit2_prob = qvm.prob_run_list(prog, qubits[2], -1)

            # 坍缩到1的概率直接当均值
            qubit0_avrage = qubit0_prob[1]

            # 这里只用第一个比特的概率
            Y_out = qubit0_avrage

            # 求后三个比特最后的状态振幅，这里还需要修改，使用的模方再开根，不含复数
            zhenfu_2 = qvm.prob_run_list(prog, [qubits[3], qubits[4], qubits[5]], -1)
            zhenfu = np.sqrt(np.array(zhenfu_2))

            # 释放局部虚拟机
            qvm.finalize()

        # 数据后处理
        if Y_out >= 0.5:
            Y_prediction = 1
        else:
            Y_prediction = 0

        Y_pre_history_test.append(Y_prediction)
        Y_true_history_test.append(Y_test[j])
        # print("预测结果：" + str(Y_prediction) +"," +"标签：" + str(Y_test[j]))

        # print("Y_prediction: " +str(Y_prediction))
        # print("Y_test[j]: " +str(Y_test[j]))

        if Y_prediction == Y_test[j]:
            a = a + 1

    print("a=" + str(a))

    accuarcy = a / test_iterations
    return accuarcy, Y_pre_history_test, Y_true_history_test


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train_1 = []
    x_train_2 = []
    for i in range(len(y_train)):
        x_train_1.append(x_train[i][[not np.all(x_train[i][j] == 0) for j in range(x_train[i].shape[0])], :])
        x_train_2.append(x_train_1[i][:, [not np.all(x_train_1[i][:, k] == 0) for k in range(x_train_1[i].shape[1])]])
    x_test_1 = []
    x_test_2 = []
    for i in range(len(y_test)):
        x_test_1.append(x_test[i][[not np.all(x_test[i][j] == 0) for j in range(x_test[i].shape[0])], :])
        x_test_2.append(x_test_1[i][:, [not np.all(x_test_1[i][:, k] == 0) for k in range(x_test_1[i].shape[1])]])
    # 过滤所有全零行，只保留非零行

    # x_train_2[x_train_2==0] = 1/2
    # x_test_2[x_test_2==0] = 1/2
    a = 8
    x_train_pca = []
    x_test_pca = []
    for i in range(len(y_train)):
        x_train_pca.append(cv.resize(x_train_2[i], dsize=(a, a)))
    for i in range(len(y_test)):
        x_test_pca.append(cv.resize(x_test_2[i], dsize=(a, a)))

    X_train_in = []
    Y_train = []
    Y_train_in = []
    for i in range(len(y_train)):
        if y_train[i] <= 1:
            X_train_in.append(x_train_pca[i])
            Y_train.append(y_train[i])
    Y_train_in = np.array(tf.one_hot(Y_train, 2))

    X_test_in = []
    Y_test = []
    Y_test_in = []
    for i in range(len(y_test)):
        if y_test[i] <= 1:
            X_test_in.append(x_test_pca[i])
            Y_test.append(y_test[i])
    Y_test_in = np.array(tf.one_hot(Y_test, 2))

    # 基本参数设置和初始化
    # iterations = len(Y_train_in)
    iterations = 200
    n = 8
    params = np.random.randn(30)
    # params = np.random.randn(30*n)
    params = list(params)  # 转换成列表用于添加元素
    params = np.array(params)  # 再转换为数组
    lr = 0.1

    print("train iterations:", iterations)
    print("block_num:", n)

    # 参数和损失函数存储初始化
    params_history = []
    loss_history = []
    zhenfu_history = []
    Y_pre_history = []
    Y_true_history = []
    gradient = []

    # 梯度下降法迭代更新参数交易量
    for i in range(iterations):
        print(i, end='|')
        X_train_n = normalize(X_train_in[i], axis=1)
        X_t_in = np.array(X_train_n)
        Y_true_history.append(Y_train_in[i])

        # 求梯度
        grad = numerical_gradient(loss, params, X_t_in, n)
        gradient.append(grad)
        # 求损失函数值（用于存储），此时预测值Y_pre_n均为差值
        LOSS, Y_prediction = loss(params, X_t_in, n)
        # 当天预测值 = 当天前一天的真实值 + 差值
        Y_pre = Y_prediction

        # 记录损失函数
        loss_history.append(LOSS)
        # print("当日交易量训练损失：\n:",loss_history)
        # 记录预测值
        Y_pre_history.append(Y_pre)
        # 梯度下降更新参数
        if i < (iterations + 1):
            params = params - lr * grad
        # 记录参数
        params_history.append(list(params))
        # print("CloseIndex " + str(i) + "loss:" + str(LOSS))

    params_A = params
    avg_loss = np.sum(loss_history[:iterations]) / iterations
    print('CloseIndex_avg_loss: ' + str(avg_loss))

    accuarcy, Y_pre_history_test, Y_true_history_test = Accuarcy(params_A, n)
    print('分类准确率：' + str(accuarcy))

end = time.time()
print('running time: %s minutes.' % ((end - start) / 60))
