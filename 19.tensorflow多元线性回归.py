import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import  Axes3D

plt.rcParams['font.sans-serif'] = ['SimHei']

x1 = tf.constant([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])

x2 = tf.constant([3.,2.,2.,3.,1.,2.,3.,2.,2.,3.,1.,1.,1.,1.,2.,2.])

y = tf.constant([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])
y = tf.reshape(y,(-1,1))

x0 =tf.ones(len(x1))
# print(y)
# print(x0)
# print(x1)
# print(x2)
# print(y)

x = tf.stack((x0,x1,x2),axis=1)

# print(x)
xt = tf.transpose(x)

xt_x_1 = tf.linalg.inv(xt@x)    #求逆矩阵

# print(xt_x_1)

xt_x_1_xt = xt_x_1@xt

w = xt_x_1_xt@y     #矩阵乘法
w = tf.transpose(w)#转置矩阵
print(w)
print(f'多元回归线性方程：y={w[0,1]}*x1+{w[0,2]}*x2+{w[0,0]}')
"""
tf.Tensor([[11.968143   0.5348893 14.331331 ]], shape=(1, 3), dtype=float32)

多元回归线性方程：y=0.5348859949724747*x1+14.331503777673149*x2+11.967290930535732
"""
# y_pred = w[0,1]*x1 + w[0,2]*x2 + w[0,0]
# y_pred = tf.expand_dims(y_pred,1)
# print(y_pred)
fig = plt.figure(figsize=(8,6))


ax3d = Axes3D(fig)
ax3d.scatter(x1,x2,y,color='b',marker='.')
ax3d.set_yticks([1,2,3])
ax3d.set_zlim3d(30,160)

ax3d.set_xlabel('area',color='r',fontsize=16)
ax3d.set_ylabel('room',color='r',fontsize=16)
ax3d.set_zlabel('price',color='r',fontsize=16)

x11,x22 = np.meshgrid(x1,x2)
print(x11)
print(x22)
y_pred = w[0,1]*x11 + w[0,2]*x22 + w[0,0]
#绘制平面图
#ax3d.plot_surface(x11,x22,y_pred,cmap='coolwarm')
ax3d.plot_wireframe(x11,x22,y_pred,cmap='coolwarm')

plt.show()