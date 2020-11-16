# -*- coding: utf-8 -*-

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
    #   E_iter              每次迭代的累积误差值，是一个列表



#sigmoid函数
def sigmoid( z ):
    return 1.0 /(1.0+np.exp(-z))