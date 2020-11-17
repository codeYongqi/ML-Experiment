# -*- coding: utf-8 -*-
import numpy as np
import math
def bp_pred(net, x):
    # 使用创建的三层的BP神经网络net进行预测，参数说明如下
    #   net                 训练好的三层BP神经网络
    #   x                   测试样本集，每一列是一个训练样本
    # 输出参数
    #   y                   BP神经网络输出
    v = net['v']
    gamma = net['gamma']
    theta = net['theta']
    w = net['w']
    alpha = np.dot(x.T,v)
    b = sigmoid(alpha-gamma,1)
    beta = np.dot(b,w)
    predictY=sigmoid(beta-theta,2)
    return predictY.T


def sigmoid(iX,dimension):#iX is a matrix with a dimension
    if dimension==1:
        for i in range(len(iX)):
            iX[i] = 1 / (1 + np.exp(-iX[i]))
    else:
        for i in range(len(iX)):
            iX[i] = sigmoid(iX[i],dimension-1)
    return iX

