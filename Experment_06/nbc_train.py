# -*- coding: utf-8 -*-

import numpy as np
def nbc_train(D, D_labels, A, lp=1):
    # 训练朴素贝叶斯分类器，参数说明如下
    #   D         训练样本集，每一行是一个训练样本
    #   D_labels  训练样本类标记，每一行是一个训练样本类标记
    #   A         训练样本属性集
    #   lp        是否进行拉普拉斯修正，1是，0否
    # 输出参数
    #   nbModel   训练好的分类器，字典类型
    nb_Model = []
    #样本个数
    train_num = D_labels.shape[0]

    #==========求类先验概率==========================
    p_good_watermelon = 0
    p_bad_watermelon  = 0
    
    num_good_watermelon = 0
    num_bad_watermelon = 0

    for i in range(train_num):
        if   D_labels[i] == 1:
            num_good_watermelon += 1
        
        elif D_labels[i] == 0:
            num_bad_watermelon  += 1

    p_good_watermelon = num_good_watermelon / train_num
    p_bad_watermelon  = num_bad_watermelon  / train_num
    #=============================================
    
    for k in range (len(A) - 1) :
        if A[k]['continue'] == 0 :
            p_con = []
            this_model = {}
            this_model['name'] = A[k]['name']
        
            for i in range (len(A[k]['values'])):
                this_id = A[k]['id'] *10 + (i+1)
                p_this_good = 0
                p_this_bad  = 0
                for j in range (train_num):
                    if (D[j][k] == this_id):
                        #print(D_labels[j])
                        if(D_labels[j] == 1):
                            p_this_good += 1
                        elif (D_labels[j] == 0):
                            p_this_bad  += 1
                
                #p_this_good = (p_this_good + 1 ) / ( num_good_watermelon + len(A[k]['values']))
                #p_this_bad  = (p_this_bad + 1)   / (num_bad_watermelon  + len(A[k]['values']))
                p_this_good = (p_this_good  ) / ( num_good_watermelon )
                p_this_bad  = (p_this_bad )   / (num_bad_watermelon  )
                p_con = [p_this_good,p_this_bad]
                this_model[this_id] = p_con
            
            this_model['continue'] = 0
            nb_Model.append(this_model)
            
        elif A[k]['continue'] == 1:
            this_model = {}
            this_good = []
            this_bad = []
            for l in range (train_num):
                if(D_labels[l] == 1):
                    this_good.append(D[l][k])
                elif(D_labels[l] == 0):
                    this_bad.append(D[l][k])

            mean_good = np.mean(this_good)
            var_good =np.std(this_good,ddof=1)

            mean_bad = np.mean(this_bad)
            var_bad = np.std(this_bad,ddof=1)
            
            this_mean = [mean_good,mean_bad]
            this_var = [var_good,var_bad]

            this_model['name'] = A[k]['name']
            this_model['mean'] = this_mean
            this_model['var'] = this_var
            this_model['continue'] = 1
            this_model['id'] = A[k]['id']
            
            nb_Model.append(this_model)

    nb_Model.append({'name' : '好瓜','p_pre':[p_good_watermelon,p_bad_watermelon]})           
    return nb_Model


    


    