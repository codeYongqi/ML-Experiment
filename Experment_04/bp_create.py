# -*- coding: utf-8 -*-
import random
import math
import numpy as np
def bp_create(x, t, hiddenLayerNum=0, eta=1.0, limit=0.001, maxNum=10, hiddenLayerFunName='sigmoid', outputLayerFunName='sigmoid'):
    # 创建一个三层的BP神经网络net，参数说明如下
    #   x                   训练样本集，每一列是一个训练样本
    #   t                   网络拟合目标，每一列对应一个训练样本
    #   hiddenLayerNum      隐层神经元个数
    #   eta                 学习率
    #	limit               累积误差阈值
    #   maxNum              连续maxNum次累积误差更新都满足阈值时停止迭代
    #   hiddenLayerFunName  隐层激活函数名称，是一个字符串，目前只能使用sigmoid函数
    #   outputLayerFunName  输出层激活函数名称，是一个字符串，目前只能使用sigmoid函数
    # 输出参数varargout
    #   net                 训练好的三层BP神经网络，是一个字典类型
    #   y                   输出层输出
    E_iter =[]             #每次迭代的累积误差值，是一个列表
    n=x.shape[0]
    d=n #输入的层个数
    l=1 #输出层个数
    hiddenLayerNum = n+1#隐层神经元个数
    #输出层的阈值
    theta = [random.random() for i in range (l)]
    #输入层的阈值
    gamma = [random.random() for i in range (hiddenLayerNum)]
    #between input side and hide nodes
    #输入到隐层的权重
    v=[[random.random() for i in range (hiddenLayerNum)]for j in range (d)]
    #隐层到输出的权重
    w=[[random.random() for i in range (l)] for j in range(hiddenLayerNum)]

    eta = 1
    sum_count=0
    t = t.T
    
    while (sum_count < maxNum) :
        #输入层*权值
        sum_count += 1
        alpha = np.dot(x.T,v)
        #得到隐层输入
        b = sigmoid(alpha-gamma,2)         
        #隐层输出*权值
        beta = np.dot(b,w)        
        #输出层输
        predictY = sigmoid(beta-theta,2)        
        #求误差
        E=sum(sum(predictY-t) **2 )/2
        #更新g
        g=predictY*(1-predictY)*(t-predictY)        
        #更新权值e
        e=b*(1-b)*( (np.dot(w ,g.T)).T )        
        #更新权值w
        w += eta*np.dot(b.T,g)
        #更新阈值
        theta -= eta*g
        #更新权值
        v += eta * np.dot(x ,e)
        #更新隐层阈值
        gamma -= eta*g
        E_iter.append(E) 
        if(E <= limit) :
            sum_count += 1
        else :    
            sum_count = 0 

    dict = {'v':v,'gamma':gamma,'theta':theta,'w':w} 
    return dict,predictY,E_iter

#iX is a matrix with a dimension
def sigmoid(iX,dimension):
    if dimension==1:
        for i in range(len(iX)):
            iX[i] = 1 / (1 + math.exp(-iX[i]))
    else:
        for i in range(len(iX)):
            iX[i] = sigmoid(iX[i],dimension-1)
    return iX

'''标准BP
sumE=0
    eta = 0.2
    flag = 1
    maxIter = 5000
    sum_count=0
    while (maxIter > 0) :
        sumE = 0
        maxIter -= 1
        for i in range (x.shape[1]):
            #输入层*权值
            alpha = np.dot(x[:,i].T,v)
            #得到隐层输入
            b = sigmoid(alpha-gamma,1)
            #隐层输出*权值
            beta = np.dot(b,w)
            #输出层输出
            predictY = sigmoid(beta-theta,1)
            #求误差
            E=sum((predictY-t[:,i]) **2 )/2
            #累计误差
            sumE += E
            #更新g
            g=predictY*(1-predictY)*(t[:,i]-predictY)
            #更新e
            e=b*(1-b)*( (np.dot(w , g.T)).T )
            #更新w
            w += eta*np.dot(b.reshape( (hiddenLayerNum,1) ),g.reshape( (1,l) ))
            #更新阈值
            theta -= eta*g
            #更新权值
            v += eta*np.dot(x[:,i].reshape((d,1)),e.reshape((1,hiddenLayerNum)))
            #更新隐层阈值
            gamma -= eta*e
        E_iter.append(sumE)


    dict = {'alpha':alpha,'gamma':gamma,'theta':theta}
    
    return dict,E_iter
'''