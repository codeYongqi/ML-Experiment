# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as scio
from nbc_train import nbc_train
from nbc_test import nbc_test
#6个离散2个连续
#data_path = 'watermelon_3.mat'
data_path = '/Users/zhuyongqi/Desktop/机器学习/数据集/watermelon_3.mat'
np.set_printoptions(suppress=True)
# 读取 mat 文件中的数据
data_dic = scio.loadmat(data_path)
data = data_dic['watermelon_3']
atts = data_dic['attributes']

# 生成样本集 D、对应的类别标记 D_labels 和属性集 A
D = data[:, 0:8]                # 训练样本集，每一行对应一个样本
D_labels = data[:, 8, None]     # 训练样本类别标记，每一行对应一个样本
A = []                          # 属性集，每个元素对应一个属性

for ii in range(9):
    att = {}
    
    if ii < 6:      # 离散属性
        att['name'] = atts[0, ii]['name'][0][0][0]
        att['values'] = atts[0, ii]['values'][0][0]
        att['continue'] = atts[0, ii]['continue'][0][0][0][0]
        att['id'] = atts[0, ii]['id'][0][0][0][0]
    elif ii < 8:    #连续属性
        att['name'] = atts[0, ii]['name'][0][0][0]
        att['continue'] = atts[0, ii]['continue'][0][0][0][0]
        att['id'] = atts[0, ii]['id'][0][0][0][0]
    else:           # 类别属性
        att['name'] = atts[0, ii]['name'][0][0][0]
        att['values'] = atts[0, ii]['values'][0][0]

    A.append(att)
#print(D)

# 测试样本                                             
testSample = np.array([11, 21, 31, 41, 51, 61, 0.697, 0.460])

#testSample = testSample[np.newaxis, :]

# 训练拉普拉斯修正的朴素贝叶斯分类器 D（17，8）D_labels(17,1)
nbModel = nbc_train(D, D_labels, A, 1)
print(nbModel)


# 测试样本分类
test_label, prob = nbc_test(nbModel, testSample)
print(test_label)
print(prob)
