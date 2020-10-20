
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
    c=sigmoid(p)
    return c

# calculate accuracy 计算准确率，一列是预测结果，一列是真实结果，结果相同则计数
def accuracy(predict_Y, Y):
    Matched = 0
    if predict_Y[0] == Y[0]:
            Matched += 1
    else:
            Matched += 0
    return Matched


path="/Users/zhuyongqi/Downloads/uci_wine.mat"
dataFile=scio.loadmat(path)
data=dataFile["wine"]
res=np.asarray(data)[:129]

X=res[...,0:12]
Y=(res[...,13:])-1


total = X.shape[0]
sum = 0
for k in range(total):
    test_index = k  # 测试集下标

    test_X = X[k]
    test_Y = Y[k]

    train_X = np.delete(X, test_index, axis=0)
    train_Y = np.delete(Y, test_index, axis=0)

    # 对率回归
    weights = grad(train_X, train_Y)
    # 统计正确率
    p= predict(test_X, weights)
    sum += accuracy(p, test_Y)

print('''留一法: ''', sum / total)

