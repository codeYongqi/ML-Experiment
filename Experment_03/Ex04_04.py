# -*- coding: utf-8 -*-

import scipy.io as scio
from tree_create_noCut import tree_create_noCut #自己编写tree_create_noCut函数
from tree_plot import tree_plot
 
data_path = '/Users/zhuyongqi/python/Experment_03/watermelon_2.mat'

# 读取 mat 文件中的数据
data_dic = scio.loadmat(data_path)
data = data_dic['watermelon_2']
atts = data_dic['attributes']

# 生成数据集 D, A, classLabel
D = data    # 训练样本集，每一行对应一个样本
A = []      # 属性集，每个元素对应一个属性

for ii in range(7):
    att = {}
    
    if ii < 6:
        att['name'] = atts[0, ii]['name'][0][0][0]
        att['values'] = atts[0, ii]['values'][0][0]
        att['continue'] = atts[0, ii]['continue'][0][0][0][0]
        att['id'] = atts[0, ii]['id'][0][0][0][0]
        A.append(att)
    else:
        att['name'] = atts[0, ii]['name'][0][0][0]
        att['values'] = atts[0, ii]['values'][0][0]
        classLabel = att    # 类别属性

print(classLabel)
# 基于基尼值进行划分选择生成决策树，不剪枝
#myTree = tree_create_noCut(D, A, classLabel)

# 绘制决策树的图
#tree_plot(myTree)