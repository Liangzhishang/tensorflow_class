import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure()
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
X = np.stack((x0,x1,x2),axis=1)
Y = price.reshape(-1,1)
# print(X)
#梯度下降法求多元线性回归

#设置超参数
learn_rate = 0.001
iter = 500

display_step = 50
np.random.seed(612)
W = np.random.randn(3,1)

mse = []

for i in range(0,iter+1):
    dL_dW = np.matmul(np.transpose(X),np.matmul(X,W)-Y) #矩阵乘法

    W = W - learn_rate*dL_dW

    PRED = np.matmul(X,W)

    Loss = np.mean(np.square(Y - PRED))/2

    mse.append(Loss)

    if i%display_step == 0:
        print(f'i:{i},loss:{Loss}')



