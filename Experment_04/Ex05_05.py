# -*- coding: utf-8 -*-

from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.io as scio
import random
from bp_create import bp_create
matplotlib.use('tkagg')
#from bp_pred import bp_pred
#关闭科学计数法
np.set_printoptions(suppress=True) 
#用来正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
#用来正常显示符号
plt.rcParams['axes.unicode_minus'] = False

data_path = '/Users/zhuyongqi/Downloads/watermelon_3.mat'

# 读取 mat 文件中的数据
data_dic = scio.loadmat(data_path)
data = data_dic['watermelon_3']


# 生成样本集和对应的类别标记
x = preprocessing.MinMaxScaler().fit_transform(data[:, 0:8]) * 2 - 1    # 对样本集的每个属性进行归一化
x = x.T                                                                 # 样本集，每一列对应一个样本
print(x.shape[1])
t = data[:, 8, None].T                                                  # 样本类别标记，每一列对应一个样本，也是神经网络拟合目标
#print(t[2])

net,E = bp_create(x,t)


fig = plt.figure(1, facecolor='white', dpi=600)    # 新建一个画布，背景设置为白色的
fig.clf()                                          # 将画图清空
ax = fig.add_subplot(1,1,1)                        # 设置一个多图展示，但是子图只有一个
plt.plot(E)
ax.set_xlabel('迭代次数')
ax.set_ylabel('累积误差值')
plt.show()
