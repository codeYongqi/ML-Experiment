# -*- coding: utf-8 -*-

import numpy as np
def nbc_test(nbModel, testSample):
    # 基于朴素贝叶斯分类器的测试样本分类
    #   nbModel       训练好的分类器，字典类型
    #   testSample    测试样本，是一个行向量
    # 输出参数
    #   test_label    测试样本类标记
    #   prob          测试样本归属于每个类的概率

    p_good = 1
    p_bad = 1
    count = 0
    for i in testSample:
        if nbModel[count]['continue'] == 0:
            p_good *= nbModel[count][i][0]
            #print(nbModel[count][i][0])
            p_bad *= nbModel[count][i][1]
            
        elif nbModel[count]['continue'] == 1:
            mean_good = nbModel[count]['mean'][0]
            mean_bad = nbModel[count]['mean'][1]

            var_good = nbModel[count]['var'][0]
            var_bad = nbModel[count]['var'][1]

            #print(p_continous(i,mean_good,var_good))
            p_good *= p_continous(i,mean_good,var_good)
            p_bad *= p_continous(i,mean_bad,var_bad)
            
        count += 1        
    
    p_good *= nbModel[count]['p_pre'][0]
    p_bad *= nbModel[count]['p_pre'][1]
    
    if (p_good > p_bad):
        test_label = '好瓜'
    else :
        test_label = '坏瓜'

    return test_label,{'好瓜概率':p_good,'坏瓜概率':p_bad}
    print(test_label)

def p_continous(x, mean,var):
    p = np.exp(-(x - mean) ** 2 * 0.5 / var ** 2) / (np.sqrt(2 * np.pi) * var)
    return p
