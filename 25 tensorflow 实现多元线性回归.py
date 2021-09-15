import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure()    #创建画布
#加载数据

area=np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])

room=np.array([3,2,2,3,1,2,3,2,2,3,1,1,1,1,2,2])

price=np.array([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])
num = len(area)

#归一化
x0 = np.ones(num)
x1 = (area - area.min())/(area.max()-area.min())
x2 = (room - room.min())/(room.max()-room.min())

# print(x1)
# print(x2)
X = np.stack((x0,x1,x2),axis=1) #堆叠
Y = price.reshape(-1,1)
# print(X)
#梯度下降法求多元线性回归

#设置超参数
learn_rate = 0.2
iter = 50

display_step = 10
np.random.seed(612)
W = tf.Variable(np.random.randn(3,1))
# print(W)
#训练模型

"""
    dL_dW = np.matmul(np.transpose(X),np.matmul(X,W)-Y) #矩阵乘法

    W = W - learn_rate*dL_dW

    PRED = np.matmul(X,W)

    Loss = np.mean(np.square(Y - PRED))/2
"""
mse = []
for i in range(0,iter+1):
    #求导数

    with tf.GradientTape(persistent=True) as tape:
        #要求导的函数：
        PRED = tf.matmul(X, W)  # 矩阵乘法
        loss = tf.reduce_mean(tf.square(Y-PRED))/2

    dL_dW = tape.gradient(loss,W)

    """
       w=w - learn_rate*dl_dw  #下一次迭代的值 = 本次迭代的值 - 步长*斜率
        
    """
    W.assign_sub(learn_rate*dL_dW) #
    mse.append(loss)

    if i%display_step == 0:
        print(f'i:{i},loss:{loss}')
        # print(dL_dW)

# plt.scatter(X,Y,color='red',label='销售记录')
# plt.scatter(X,PRED,color = 'b',label='梯度下降法')
# plt.plot(X,PRED,color='b')

del tape
# plt.plot(mse)
plt.show()
"""
i:0,loss:9187.70331286175
i:50,loss:9187.70331286175
i:100,loss:9187.70331286175
i:150,loss:9187.70331286175
i:200,loss:9187.70331286175
i:250,loss:9187.70331286175
i:300,loss:9187.70331286175
i:350,loss:9187.70331286175
i:400,loss:9187.70331286175
i:450,loss:9187.70331286175
i:500,loss:9187.70331286175
"""