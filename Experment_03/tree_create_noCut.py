import scipy.io as scio
import numpy as np

class_name='好瓜'
D_keys = {
	'色泽': ['青绿', '乌黑', '浅白'], 
	'根蒂': ['蜷缩', '硬挺', '稍蜷'], 
	'敲声': ['清脆', '沉闷', '浊响'], 
	'纹理': ['稍糊', '模糊', '清晰'], 
	'脐部': ['凹陷', '稍凹', '平坦'], 
	'触感': ['软粘', '硬滑'], 
}

def choose_largest_example(D):
	Count = D[class_name].value_counts()
	Max, Max_key = -1, None
	for key, value in Count.items():
		if value > Max:
			Max = value
			Max_key = key

	return Max_key
 
def same_value(D, A):
	for key in A:
		if len(D[key].value_counts()) > 1:
			return False

	return True

def Gini(D):
    Sum=0
    Total=D.shape[0]
    Count=D[class_name].value_counts()
    for key,value in Count.items():
        prob=value/Total
        Sum+=prob**2
    return 1- Sum    

def calc_Gini_index(D, key):
	Sum, D_size = 0, D.shape[0]
	for value in D_keys[key]:
		Dv = D.loc[D[key]==value]
		Dv_size = Dv.shape[0]

		Sum += (Dv_size/D_size) * Gini(Dv)

	return Sum

def choose_best_attribute(D, A):
	min_Gini_index = 999
	for key in A:

		Gini_index = calc_Gini_index(D, key)

		if min_Gini_index > Gini_index:
			min_Gini_index = Gini_index
			best_attr = key

	return best_attr

def tree_create_noCut(D, A, D_test):
    
	Count = D[class_name].value_counts()
	
	if len(Count) == 1:
		return D[class_name].values[0]
	
	if len(A)==0 or same_value(D, A):
		return choose_largest_example(D)

	node = {}

	best_attr = choose_best_attribute(D, A)

	new_A = [key for key in A if key != best_attr]

	for value in D_keys[best_attr]:
		Dv = D.loc[D[best_attr]==value]
		Dv_test = D_test.loc[D_test[best_attr]==value]

		if Dv.shape[0] == 0:
			node[value] = choose_largest_example(D)
		else:
			node[value] = tree_create_noCut(Dv, new_A, Dv_test)

	# plotTree.plotTree(Tree)
	return {best_attr: node}