
import numpy as np
import scipy.io as scio

np.set_printoptions(suppress=True) 
def sigmoid( z ):
    return 1.0 /(1.0+np.exp(-z))
    
def grad(train_X, labels, iters=2000):
    m, n = train_X.shape
    # 步长alpha
    alpha = 0.05
    # 初始化权重，全设为1
    weights = np.ones((n, 1))

    # 2000次迭代
    for k in range(iters):
        # 沿着梯度方向，向前移动，并更新权重
        P = sigmoid(train_X.dot(weights))
        error = labels - P
        weights += alpha * np.dot(train_X.T, error)

    return weights

def predict(test_X, weights):
    m = test_X.shape[0]
    #由sigmoid函数的性质，z = w * x , z大于0时，sigmoid(Z)>0.5 即预测为1，反之预测为0 
    p = np.dot(test_X, weights)
    for k in range(m):
        p[k]=sigmoid(p[k])
    return p


# calculate accuracy 计算准确率，一列是预测结果，一列是真实结果，结果相同则计数
def accuracy(predict_Y, Y):
    m, n = Y.shape
    Matched = 0
    for k in range(m):
        if predict_Y[k] == Y[k]:
            Matched += 1
        else:
            Matched += 0
    return Matched / m


path="/Users/zhuyongqi/Downloads/uci_wine.mat"
dataFile=scio.loadmat(path)
data=dataFile["wine"]
res=np.asarray(data)[:129]

X=res[...,0:12]
Y=(res[...,13:])-1


total = X.shape[0]
num_split = int(total / 10)
sum = 0

for k in range(10):

    #选择测试集的下标
    test_index = range(k * num_split , (k+1) * num_split)
    
    test_X = X[test_index]
    test_Y = Y[test_index]
    
    train_X = np.delete(X,test_index,axis=0)
    train_Y = np.delete(Y,test_index,axis=0)
    
    #求对率回归最优参数
    weights = grad(train_X,train_Y)
    #print(weights)
    #统计每次组的正确率
    p = predict(test_X,weights)
    sum += accuracy(p,test_Y)
    #result += predict(test_X,weights)==test_Y ? 1:0

#正确次数 / 验证总次数 = 准确率
print('''10折交叉验证 ''',sum/10)


