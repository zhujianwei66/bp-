# -*- coding: utf-8 -*-
"""
@File        :  getdata.py
@Time        :  2021/9/20 20:00
@Author      :  JianweiZhu
@Contact     :  jianwei_zhu959@163.com
@Software    : PyCharm
@Description :
"""


import numpy as np
import cv2 as cv

def change(m):
    """把一个数据矩阵变成一行"""
    M = []
    for i in range(m.shape[1]):
        temp = m[:][i]
        M = M + list(temp)
    return M


def loaddataset(path):
    """读取数据集"""

    label = np.zeros((400, 40))  # 标签
    train = []  # 样本矩阵
    flag = 0
    for i in range(1, 41):
        for j in range(1, 11):
            img = cv.imread(path + f"s{i}\\{j}.pgm", 0)
            train.append(change(img))
            label[flag, i - 1] = 1
            flag = flag + 1
    # 二维数组转np数组
    X = np.array(train)
    return X, label


def mypca(data, n):
    """data:需要降维的矩阵
    n取对应特征值最大的特征向量个数
    """
    # 计算原始数据中每一列的均值，axis=0按列取均值
    mean = np.mean(data, axis=0)
    # 数据中心化，使每个feature的均值为0
    zeroCentred_data = data - mean
    # 计算协方差矩阵，rowvar=False表示数据的每一列代表一个feature
    covMat = np.cov(zeroCentred_data, rowvar=False)
    # 计算协方差矩阵的特征值和特征向量
    featValue, featVec = np.linalg.eig(covMat)
    # 将特征值按从小到大排序，index是对应原featValue中的下标
    index = np.argsort(featValue)
    # 取最大的n个特征值在原featValue中的下标
    n_index = index[-n:]
    # 取最大的两维特征值对应的特征向量组成映射矩阵
    n_featVec = featVec[:, n_index]
    # 降维后的数据
    low_dim_data = np.dot(zeroCentred_data, n_featVec)
    return low_dim_data

def read():
    path = "D:\\MyWorkPlace\\pattern recognition\\ORL_Faces\\"
    # 读取样本矩阵和期望输出
    dataset, labelset = loaddataset(path)
    # 对样本矩阵进行pca降维
    dataset_pca = mypca(dataset, 71).astype(np.float64)
    # 保存降维后矩阵
    np.savez("mydata.npz", dataset_pca=dataset_pca, labelset=labelset)

read()
