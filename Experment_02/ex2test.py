import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('tkagg')

m=20

X0=np.ones([m,1])
print(X0)
X1=np.arange(1,m+1).reshape(m,1)
print(X1)
X=np.hstack([X0,X1])
print(X)


y=np.array([
    3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    11, 13, 13, 16, 17, 18, 17, 19, 21
]).reshape(m,1)

alpha=0.01

def error_function(theta,X,y):
    diff=np.dot(X,theta) -y
    return (1.0/2*m)*np.dot(diff.T,diff)

def gradient_function(theta,X,y):

    diff=np.dot(X,theta)-y
    return (1.0/m)*np.dot(X.T,diff)
    
def graditent_descent(X,y,alpha):
    theta=np.array([1,1]).reshape(2,1)
    gradient=gradient_function(theta,X,y)
    while not np.all(np.absolute(gradient)<=1e-5):
        theta=theta-alpha*gradient
        gradient=gradient_function(theta,X,y)
    return theta        


optimal=graditent_descent(X,y,alpha)
print('optimal:',optimal)
plt.figure(2)
plt.plot(X1,y,"o")

print("error function",error_function(optimal,X,y)[0,0]) 
c=np.dot(X,optimal)
plt.plot(X1,c)  
plt.show()