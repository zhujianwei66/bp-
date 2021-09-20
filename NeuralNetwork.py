# -*- coding: utf-8 -*-
"""
@File        :  NeuralNetwork.py
@Author      :  JianweiZhu
@Contact     :  jianwei_zhu959@163.com
@Software    : PyCharm
@Description :
"""
import numpy as np
import cv2 as cv
import time


# 在bp神经网络的基础上添加了冲量项，并设置自适应的学习率
# 自适应学习率函数
def RateAdaptive(eta, k, c):
    if k < 1:
        return 1.02 * eta
    if k > c:
        return 0.98 * eta
    return eta

# x为输入层神经元个数，y为隐层神经元个数，z输出层神经元个数
def parameter_initialization(x, y, z):
    # 隐层阈值
    value1 = np.zeros((1, y)).astype(np.float64)
    # 输出层阈值
    value2 = np.zeros((1, z)).astype(np.float64)
    # 输入层与隐层的连接权重
    weight1 = np.random.randn(x, y) / np.sqrt(x / 2).astype(np.float64)
    # 隐层与输出层的连接权重
    weight2 = np.random.randn(y, z) / np.sqrt(y / 2).astype(np.float64)
    return weight1, weight2, value1, value2


def sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))  # 优化sigmoid函数，避免溢出


def relu(z):  # RELU激活函数，暂时未用到
    return np.maxinum(0, z)


'''
weight1:输入层与隐层的连接权重
weight2:隐层与输出层的连接权重
value1:隐层阈值
value2:输出层阈值
'''


def trainning(dataset, labelset, weight1, weight2, value1, value2, eta, weight1_change_last, weight2_change_last):
    # eta学习效率
    i = 0
    while i < len(dataset):
        # 输入数据
        inputset = np.mat(dataset[i]).astype(np.float64)
        # 数据标签
        outputset = np.mat(labelset[i]).astype(np.float64)
        # 隐层输入
        input1 = np.dot(inputset, weight1).astype(np.float64)
        # 隐层输出
        output2 = sigmoid(input1 - value1).astype(np.float64)
        # 输出层输入
        input2 = np.dot(output2, weight2).astype(np.float64)
        # 输出层输出
        output3 = sigmoid(input2 - value2).astype(np.float64)

        # 更新公式由矩阵运算表示
        a = np.multiply(output3, 1 - output3)
        g = np.multiply(a, outputset - output3)
        b = np.dot(g, np.transpose(weight2))
        c = np.multiply(output2, 1 - output2)
        e = np.multiply(b, c)

        # 计算偏置值增量，权值增量
        value1_change = -eta * e
        value2_change = -eta * g
        weight1_change = eta * np.dot(np.transpose(inputset), e)
        weight2_change = eta * np.dot(np.transpose(output2), g)

        # 临时更新参数
        value1_tmp = value1 + value1_change
        value2_tmp = value2 + value2_change
        weight1_tmp = weight1 + 0.6 * weight1_change_last + eta * 0.4 * weight1_change
        weight2_tmp = weight2 + 0.6 * weight2_change_last + eta * 0.4 * weight2_change

        # 修改后误差值平方和
        input1_after = np.dot(inputset, weight1_tmp).astype(np.float64)
        output2_after = sigmoid(input1_after - value1_tmp).astype(np.float64)
        input2_after = np.dot(output2_after, weight2_tmp).astype(np.float64)
        output3_after = sigmoid(input2_after - value2_tmp).astype(np.float64)
        after = np.dot(output3_after - outputset, (output3_after - outputset).T).astype(np.float64)
        # 修改前误差值平方和
        before = np.dot(output3 - outputset, (output3 - outputset).T).astype(np.float64)

        k = before / after
        eta = RateAdaptive(eta, k, 2)  # 修正学习速率

        if k > 2 and eta > 0.01 and eta < 1:  # 限制学习效率（0.01~1）
            continue
        i = i + 1
        # 保留权值、偏置值修改
        value1 = value1_tmp
        value2 = value2_tmp
        weight1 = weight1_tmp
        weight2 = weight2_tmp
        # 冲量项
        weight1_change_last = weight1_change
        weight2_change_last = weight2_change

    return weight1, weight2, value1, value2, eta, weight1_change_last, weight2_change_last


def testing(dataset, labelset, weight1, weight2, value1, value2):
    # 记录预测正确的个数
    rightcount = 0
    for i in range(len(dataset)):
        # 计算每一个样例通过该神经网路后的预测值
        inputset = np.mat(dataset[i]).astype(np.float64)
        outputset = np.mat(labelset[i]).astype(np.float64)
        output2 = sigmoid(np.dot(inputset, weight1) - value1)
        output3 = sigmoid(np.dot(output2, weight2) - value2)

        # 确定其预测标签
        if labelset[i, np.argsort(output3)[0, -1]] == 1:
            rightcount += 1
        else:
            print("预测为%2s   实际为%2s" % (np.argsort(output3)[0, -1], np.argsort(labelset[i])[-1]))  # 预测错误则输出
            continue
        # 输出预测结果
    # 返回正确率
    return rightcount / len(dataset)


def test():
    # 获取测试集
    mydata = np.load("mydata.npz")
    dataset_pca = mydata['dataset_pca']
    labelset = mydata['labelset']
    testset = []
    testlabel = []
    for i in range(40):
        for j in range(3):
            testset.append(dataset_pca[i * 10 + j + 7])
            testlabel.append(labelset[i * 10 + j + 7])
    testset = np.array(testset)
    testlabel = np.array(testlabel)

    # 读取网络
    net = np.load("net.npz")
    # 权值、偏置值、学习速率
    weight1 = net['weight1']
    weight2 = net['weight2']
    value1 = net['value1']
    value2 = net['value2']

    rate = testing(testset, testlabel, weight1, weight2, value1, value2)
    print("正确率:%0.4f" % (rate))


def train(time):
    # 获取训练集
    mydata = np.load("mydata.npz")
    dataset_pca = mydata['dataset_pca']
    labelset = mydata['labelset']
    trainset = []
    trainlabel = []
    for i in range(40):
        for j in range(7):
            trainset.append(dataset_pca[i * 10 + j])
            trainlabel.append(labelset[i * 10 + j])
    trainset = np.array(trainset)
    trainlabel = np.array(trainlabel)

    # 初始化神经网络
    weight1, weight2, value1, value2 = parameter_initialization(len(trainset[0]), 100, 40)
    weight1_change_last = np.zeros(np.shape(weight1)).astype(np.float64)
    weight2_change_last = np.zeros(np.shape(weight2)).astype(np.float64)
    eta = 0.5
    np.savez("net.npz", weight1=weight1, weight2=weight2, value1=value1, value2=value2, eta=eta,
             weight1_change_last=weight1_change_last, weight2_change_last=weight2_change_last, rate=0)
    rate_last = 0
    # 训练次数time
    for i in range(time):
        # 返回权重1、权重2，偏置1，偏置2，学习效率，权值增量1，权值增量2，
        weight1, weight2, value1, value2, eta, weight1_change_last, weight2_change_last = trainning(trainset,
                                                                                                    trainlabel, weight1,
                                                                                                    weight2, value1,
                                                                                                    value2, eta,
                                                                                                    weight1_change_last,
                                                                                                    weight2_change_last)
        print(f"\r迭代次数:{i + 1}", end=" ")
    # 保存数据
    np.savez("net.npz", weight1=weight1, weight2=weight2, value1=value1, value2=value2)  # 保存数据



if __name__ == '__main__':
    start = time.time()
    train(200)  # 训练神经网络经过多次实验，100次已经有足够的效果
    end = time.time()
    print("\n训练用时:%0.2fs" % (end - start))
    test()  # 测试神经网络