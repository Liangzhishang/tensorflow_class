import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import  Axes3D

x1=np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])

x2=np.array([3,2,2,3,1,2,3,2,2,3,1,1,1,1,2,2])

y=np.array([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])

# print(np.shape(x1))
# print(np.shape(x2))
# print(np.shape(x3))

x0=np.ones(len(x1))

x=np.stack((x0,x1,x2),axis=1)

# print(x)

Y=np.array(y).reshape(-1,1)
# print(Y)

#计算x的转置
xt = np.transpose(x)

xtx_1 = np.linalg.inv(np.matmul(xt,x))

xtx_1_xt = np.matmul(xtx_1,xt)

w=np.matmul(xtx_1_xt,Y)
w=w.reshape(-1)
print(w)

print(f'多元回归线性方程：y={w[1]}*x1+{w[2]}*x2+{w[0]}')


