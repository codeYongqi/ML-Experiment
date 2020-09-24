
import scipy.io as sico
import numpy as np
import scipy.misc
from PIL import Image

dataFile='/Users/zhuyongqi/Downloads/yalemat.mat'
data=sico.loadmat(dataFile)

a=data['yalemat']

##F列优先
c=a[:,0].reshape((32,32),order='F')

im = Image.fromarray(c).save("out.jpeg")

