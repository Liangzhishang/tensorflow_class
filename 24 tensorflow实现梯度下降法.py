
#tensorflow实现一元线性回归
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure()
#加载数据

x=np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])

#x2=np.array([3,2,2,3,1,2,3,2,2,3,1,1,1,1,2,2])

y=np.array([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])

#设置超参数
learn_rate = 0.00001
iter = 100

dis_play_step = 10

#设置模型参数初值
np.random.seed(612)
w = tf.Variable(np.random.randn())#一个随机数
b = tf.Variable(np.random.randn())

#训练模型
mse=[]
for i in range(0,iter+1):
    #求梯度
    with tf.GradientTape() as tape:
        prep = w*x + b
        loss = tf.reduce_mean(tf.square(y - prep)) / 2 #

    mse.append(loss)

    dl_dw,dl_db = tape.gradient(loss,[w,b])

    """
    w=w - learn_rate*dl_dw  #下一次迭代的值 = 本次迭代的值 - 步长*斜率
    b=b - learn_rate*dl_db 
    """
    w.assign_sub(learn_rate*dl_dw)
    b.assign_sub(learn_rate*dl_db)

    if i%dis_play_step == 0:
        print(f'i:{i},loss:{loss},w:{w},b:{b}')

# plt.scatter(x,y,color='red',label='销售记录')
# plt.scatter(x,prep,color = 'b',label='梯度下降法')
# plt.plot(x,prep,color='b')

del tape
plt.plot(mse)
plt.show()


