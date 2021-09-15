import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

x = tf.constant([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
y = tf.constant([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])

#y=wx+b 计算w和b

meanx = tf.reduce_mean(x)#
meany = tf.reduce_mean(y)#平均值

# print(meanx)
# print(meany)

# print((x-meanx)*(y-meany))#张量中每一项都减去平均值
xy = (x-meanx)*(y-meany)
# print(xy)

x2 = (x-meanx)*(x-meanx)
# print(x2)

sumxy = tf.reduce_sum(xy)
sumx = tf.reduce_sum(x2)
# print(sumxy)
# print(sumx)

w = sumxy/sumx
# print(w)

b = meany - w*meanx
# print(b)

# out = w * x_in + b
x_text = np.array([128.15,45.00,141.43,106.27,99,53.84,85.36,70.00])
print( )
y_prep=w * x_text + b
#数据和模型可视化

# plt.figure()
# plt.scatter(x,y)#绘制数据点
plt.scatter(x,y,color = 'red',label = '销售记录')
plt.scatter(x_text,y_prep,color='blue',label ='拟合曲线',linewidth = 2)

plt.plot(x_text,y_prep,color = 'blue',label = '拟合曲线',linewidth = 2)


plt.xlim((40,160))
plt.ylim((40,150))

plt.show()