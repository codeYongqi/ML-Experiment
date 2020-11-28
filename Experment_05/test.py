import scipy.io as sico
import numpy as np
import cvxopt
from numpy import linalg
from sklearn import preprocessing

dataFile='/Users/zhuyongqi/Desktop/机器学习/数据集/watermelon_3a.mat'
data=sico.loadmat(dataFile)
#np.set_printoptions(suppress=True) 
a = data['watermelon_3a']
b=a[:,[0,1]]
features=a[:,2]
features=features*2-1
# 进行归一化处理
X_scaled = preprocessing.scale(b)

def gaussian_kernel(x, y, sigma):
    #return np.exp(-(sum((x-y)**2))/ (2 * (sigma ** 2)))
    return np.exp(-sigma*(sum((x-y)**2)))
n_samples, n_features = X_scaled.shape

Min_err_rate = 1
min_sigma = 0
end_result = np.empty([1,n_samples])
for sigma in range(1, 10000):
    sigma = 0.0001 * sigma
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = gaussian_kernel(X_scaled[i], X_scaled[j],sigma)

    E = np.ones([n_samples,1])
    Y = np.zeros([n_samples,n_samples])
    y = np.ones([n_samples,1])

    for i in range (n_samples):
        y[i] = features[i]

    y = y.reshape(1,-1)   
    Y = np.diag(features)

    q = E

    P =np.dot(np.dot(Y , K)  , Y)
    H = np.zeros([n_samples,1])

    G = -1 * np.ones([n_samples])
    G = np.diag(G)
    P = P.astype(np.double)

    P = cvxopt.matrix(P,(17,17))
    q = cvxopt.matrix(q,(17,1))
    G = cvxopt.matrix(G,(17,17))
    A = cvxopt.matrix(y,(1,17))
    h = cvxopt.matrix(H,(17,1))
    b = cvxopt.matrix(0.0)

    sol = cvxopt.solvers.qp(P,q,G,h,A,b) 
    res = sol['x']

    S = []
    for i in range(n_samples):
        if(res[i]>0):
            S.append(i)
    if(len(S)==0):
        continue
    temp = 0
    b = 0

    for s in  S:
        for i in  S:
            temp += res[i] * y.T[i] * K[i][s]
        b += 1 / y.T[s] - temp
    b /= len(S)
    print(b)

    err = 0
    result = 0
    res_pre=[]
    for j in range(n_samples):
        result = 0
        for i in range(n_samples):
            result += res[i] * y.T[i] * gaussian_kernel(X_scaled[j],X_scaled[i],sigma)
        result = result+b

        if result > 0:
            result = 1
        elif result < 0:
            result = -1
        res_pre.append(result)

    for i in range(n_samples):
        if res_pre[i]!=y.T[i]:
            err +=1
    err_rate = err / n_samples 
    if Min_err_rate > err_rate:
        Min_err_rate = err_rate
        min_sigma = sigma
        end_res = res
        end_b = b
        end_result = res_pre
    print(Min_err_rate)
    print(end_result)    

    