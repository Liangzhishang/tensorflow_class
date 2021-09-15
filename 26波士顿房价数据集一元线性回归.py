#加载数据集

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

boston_housing = tf.keras.datasets.boston_housing
(train_x,train_y),(test_x,test_y) = boston_housing.load_data()

print(tf.shape(train_x))
print(tf.shape(train_y))
print(tf.shape(test_x))
print(tf.shape(test_y))

x_train = train_x[:,5] #所有的房间数
y_train = train_y

x_test = test_x[:,5]
y_test = test_y

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

#设置超参数
learn_rate = 0.04
iter=5000
display_step = 200

#设置模型参数初始值
np.random.seed(612)
w = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())

# print(w)
# print(b)

#训练模型：
mse_train = []
mse_test = []

for i in range(0,iter+1):
    #需要求导的函数
    with tf.GradientTape(persistent=True) as tape:
        prep_train = w*x_train + b
        loss_train = tf.reduce_mean(tf.square(y_train-prep_train))/2

        prep_test = w*x_test + b
        loss_test = tf.reduce_mean(tf.square(y_test-prep_test))/2

    #求导
    dloss_train_dw = tape.gradient(loss_train,w)
    dloss_test_dw = tape.gradient(loss_test,w)

    dloss_train_db = tape.gradient(loss_train, b)
    dloss_test_db = tape.gradient(loss_test, b)

    # w = w-learn_rate*dloss_train_dw
    # b = w-learn_rate*dloss_train_db

    w.assign_sub(learn_rate*dloss_train_dw)
    b.assign_sub(learn_rate*dloss_train_db)

    mse_train.append(loss_train)


    if i % display_step == 0:
        print(f'i:{i},loss:{loss_train},w:{w},b:{b}')


#可视化
plt.figure(figsize=(15,10))
plt.subplot(221)
plt.scatter(x_train,y_train,color='blue',linewidth=1.5,label="train loss")
plt.plot(x_train,prep_train,color = 'red',linewidth=3,label='test loss')


plt.subplot(222)
plt.plot(mse_train,color='blue',linewidth=1.5,label="train loss")
plt.plot(mse_test,color = 'red',linewidth=3,label='test loss')

plt.subplot(223)
plt.plot(y_train,color='blue',linewidth=1.5,label="train loss")
plt.plot(prep_train,color = 'red',linewidth=3,label='test loss')

plt.subplot(224)
plt.plot(y_test,color='blue',linewidth=1.5,label="train loss")
plt.plot(prep_test,color = 'red',linewidth=3,label='test loss')

plt.show()

del tape
"""
i:0,loss:23.641782760620117,w:0.9456712603569031,b:0.9456712603569031
i:200,loss:17.3513126373291,w:0.694052517414093,b:0.694052517414093
i:400,loss:17.3513126373291,w:0.694052517414093,b:0.694052517414093
i:600,loss:17.3513126373291,w:0.694052517414093,b:0.694052517414093
i:800,loss:17.3513126373291,w:0.694052517414093,b:0.694052517414093
i:1000,loss:17.3513126373291,w:0.694052517414093,b:0.694052517414093
i:1200,loss:17.3513126373291,w:0.694052517414093,b:0.694052517414093
i:1400,loss:17.3513126373291,w:0.694052517414093,b:0.694052517414093
i:1600,loss:17.3513126373291,w:0.694052517414093,b:0.694052517414093
i:1800,loss:17.3513126373291,w:0.694052517414093,b:0.694052517414093
i:2000,loss:17.3513126373291,w:0.694052517414093,b:0.694052517414093
"""