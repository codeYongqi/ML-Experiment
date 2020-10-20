def predict(test_X, weights):
    m = test_X.shape[0]
    #由sigmoid函数的性质，z = w * x , z大于0时，sigmoid(Z)>0.5 即预测为1，反之预测为0 
    p = np.dot(test_X, weights)
    c=sigmoid(p)
    return c
