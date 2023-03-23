import math as m
import numpy as np
import pandas as pd
import time
from pyqpanda import *
from pyvqnet.nn.module import Module
from pyvqnet.qnn.quantumlayer import QuantumLayer
from pyvqnet.optim import adam
from pyvqnet.nn.loss import MeanSquaredError
from pyvqnet.tensor import tensor
from pyvqnet.tensor import QTensor
import tensorflow as tf
from tensorflow.keras.utils import normalize
from sklearn.decomposition import PCA
import cv2 as cv
from sklearn import decomposition

start = time.time()
Amplitude = np.array([1, 0, 0, 0, 0, 0, 0, 0])  # 定义振幅为全局变量
'''Step1 量子线路定义'''


# 数据输入U_in矩阵
def U_in(qubits, X_t):
    circuit = create_empty_circuit()
    theta_in = m.acos(X_t.item())
    circuit << RY(qubits[0], theta_in) \
    << RY(qubits[1], theta_in) \
    << RY(qubits[2], theta_in)
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
    circuit << U_theta(qubits, params1) \
    << H_X(qubits, params2) \
    << H_ZZ(qubits, params3)
    return circuit


# 批量读取数据
def get_minibatch_data(x_data, true, batch_size):
    for i in range(0, x_data.shape[0] - batch_size + 1, batch_size):
        idxs = slice(i, i + batch_size)
        yield x_data[idxs], true[idxs]  # yeild  类似于return，返回后交出CPU使用权


# 搭建完整的量子线路
def QCircuit(input, weights, qlist, clist, machine):
    global Amplitude
    x = input
    params = weights.squeeze()
    prog = QProg()
    circuit = create_empty_circuit()
    circuit << amplitude_encode([qlist[0], qlist[1], qlist[2]], x)  # 数据输入
    circuit << amplitude_encode([qlist[3], qlist[4], qlist[5]], Amplitude)
    circuit << QRNN_VQC(qlist, params[0: 30])
    prog << circuit
    qubit0_prob = machine.prob_run_dict(prog, qlist[0], -1)
    qubit1_prob = machine.prob_run_list(prog, qlist[1], -1)
    qubit2_prob = machine.prob_run_list(prog, qlist[2], -1)
    Amplitude_2 = machine.prob_run_list(prog, [qlist[3], qlist[4], qlist[5]], -1)
    Amplitude = np.sqrt(np.array(Amplitude_2))
    prob = list(qubit0_prob.values())
    return prob


'''Step2 定义一个继承于Module的机器学习模型类'''
param_num = 30  # 待训练参数个数
qbit_num = 6  # 量子计算模块量子比特数


class QRNNModel(Module):
    def __init__(self):
        super(QRNNModel, self).__init__()
        # 使用QuantumLayer类，可以把带训练参数的量子线路纳入VQNet的自动微分的训练流程中
        self.pqc = QuantumLayer(QCircuit, param_num, "cpu", qbit_num)

    # 定义模型前向函数
    def forward(self, X_t):
        global Amplitude
        Amplitude = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        xin = X_t
        # xin = normalize(X_t, axis=1)

        for i in range(X_t.shape[1]):
            input = tensor.unsqueeze(xin[i])  # 输入必须满足[batch_size, n]的格式

            x = self.pqc(input)[0,1]  # 坍缩到1的概率直接当均值，这里只用第一个比特的概率
            # print(Amplitude)
        return x


'''Step3 模型的训练'''


def train(data):

    x_train = []
    y_train = []

    x_train = np.array(X_train_in)
    y_train = np.array(Y_train)
    x_train = x_train.reshape(-1, n)
    # 3. 模型的训练
    batch_size = 1
    epoch = 300
    print("start training...........")
    for i in range(epoch):
        model.train()
        count = 0
        loss = 0
        for data, true in get_minibatch_data(x_train, y_train, batch_size):
            data, true = QTensor(data), QTensor(true)
            optimizer.zero_grad()  # 优化器中缓存梯度清零
            output = model(data)  # 模型前向计算
            losss = MseLoss(true, output)  # 损失函数计算
            # print(data, true, output)
            losss.backward()  # 损失反向传播
            optimizer._step()  # 优化器参数更新
            loss += losss.item()
            count += batch_size
        print(f"epoch:{i}, train_loss:{loss / count}\n")
    Param = np.array(model.pqc.parameters())[0].reshape((-1, 1)).to_numpy().squeeze()
    return Param


'''Step3 性能测试'''


def Accuarcy(params, zhibiao, n):

    
    # test_iterations = len(Y_test)
    test_iterations = 100
    print("test iterations = ", test_iterations)

    #  这里两个用于记录测试集的真实值和预测值
    _ = []
    _ = []
    
    a = 0

    Ei_2_sum = 0  # 误差平方和初始化
    for j in range(test_iterations):
        zhenfu = np.array([1, 0, 0, 0, 0, 0, 0, 0 ])
        X_test_n = normalize(X_test_in[j], axis=1)
        X_t = np.array(X_test_n)

        for i in range(n):
            qvm = CPUQVM()  # 建立一个局部的量子虚拟机
            qvm.init_qvm()  # 初始化量子虚拟机
            qubits = qvm.qAlloc_many(6)
            # cbits = qvm.cAlloc_many(6)
            prog = QProg()
            circuit = create_empty_circuit()

            circuit << amplitude_encode([qubits[0], qubits[1], qubits[2]], X_t[i])  # 数据输入
            circuit << amplitude_encode([qubits[3], qubits[4], qubits[5]], zhenfu)  # 后三个比特的编码
            circuit << QRNN_VQC(qubits, params[0: 30])
            prog << circuit

            qubit0_prob = qvm.prob_run_list(prog, qubits[0], -1)
            qubit1_prob = qvm.prob_run_list(prog, qubits[1], -1)
            qubit2_prob = qvm.prob_run_list(prog, qubits[2], -1)

            # 坍缩到1的概率直接当均值
            qubit0_avrage = qubit0_prob[1]

            # 求后三个比特最后的状态振幅，这里还需要修改，使用的模方再开根，不含复数
            zhenfu_2 = qvm.prob_run_list(prog, [qubits[3], qubits[4], qubits[5]], -1)
            zhenfu = np.sqrt(np.array(zhenfu_2))

            # 释放局部虚拟机
            qvm.finalize()

        # 数据后处理
        if qubit0_avrage <= 0.5:
            Y_prediction = 0
        else:
            Y_prediction = 1

        # _.append(Y_prediction * sum + Y_tmin)
        _.append(Y_prediction)
        _.append(Y_test[j])
        
        if Y_prediction == Y_test[j]:
            a = a + 1
            
    print("a=" +str(a))
    accuarcy = a/test_iterations
    return accuarcy, _, _


if __name__ == '__main__':
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    #去除全0行
    x_train_1 = []
    x_train_2 = []
    for i in range(len(y_train)):
        x_train_1.append(x_train[i][[not np.all(x_train[i][j] == 0) for j in range(x_train[i].shape[0])],:])
        x_train_2.append(x_train_1[i][:,[not np.all(x_train_1[i][:,k] == 0) for k in range(x_train_1[i].shape[1])]])
        
    x_test_1 = []
    x_test_2 = []
    for i in range(len(y_test)):
        x_test_1.append(x_test[i][[not np.all(x_test[i][j] == 0) for j in range(x_test[i].shape[0])],:])
        x_test_2.append(x_test_1[i][:,[not np.all(x_test_1[i][:,k] == 0) for k in range(x_test_1[i].shape[1])]])
    
    x_train_2[x_train_2==0] = 1
    x_test_2[x_test_2==0] = 1
    
    a=8
    
    x_train_pca = []
    for i in range(len(y_train)):
        x_train_pca.append(cv.resize(x_train_2[i], dsize = (a, a)))
    x_test_pca = []
    for i in range(len(y_test)):
        x_test_pca.append(cv.resize(x_test_2[i], dsize = (a, a)))

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
    
    n = 8
    
    model = QRNNModel()
    optimizer = adam.Adam(model.parameters(), lr=0.01)
    MseLoss = MeanSquaredError()
    
    param = train(X_train_in)
    accuarcy, _, _ = Accuarcy(param, X_test_in, n)
    print('相对湿度精度为：' + str(accuarcy))
    # np.savetxt(f"./QRNN_best_params/Relative Humidity.txt", param)
    
