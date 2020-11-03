import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('tkagg')

x=np.arange(-100,100)
y=x*x
plt.plot(x,y)

plt.show()