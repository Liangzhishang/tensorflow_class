import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure()

x=np.array([1.,2.,3.,4.])
w=tf.Variable(1.)
b=tf.Variable(1.)

# print(f'{x} {w} {b}')
x=np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])

# y=np.array([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])

y=np.array([1,1,0,1,0,1,1,0,0,1,0,0,0,0,0,0])


# plt.scatter(x,y)
# print(np.shape(x))
# print(np.shape(y))
# print(1-y)
# print(1-pred)

x_train = x - np.mean(x)
y_train = y

#设置超参数
learn_rate = 0.005
iter=5
display_step = 1

#设置模型初始值
np.random.seed(612)
w = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())


x_ = range(-80,80)
y_ = 1/(1+tf.exp(-(w*x_+b)))

# plt.plot(x_,y_)
#训练模型
cross_train = []
acc_train = []

for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        pred_train = 1/(1+tf.exp(-(w*x_train+b)))
        loss_train = -tf.reduce_mean(y_train*tf.math.log(pred_train)+(1-y_train)*tf.math.log(1-pred_train))
        Accuracy_train = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred_train<0.5,0,1),y_train),tf.float32))


    cross_train.append(loss_train)
    acc_train.append(Accuracy_train)

    dL_dw,dL_db = tape.gradient(loss_train,[w,b])

    w.assign_sub(learn_rate*dL_dw)
    b.assign_sub(learn_rate*dL_db)

    if i%display_step==0:
        print(f'i:{i},loss_train:{loss_train},accuracy:{Accuracy_train}')
        y_ = 1 / (1 + tf.exp(-(w * x_ + b)))

        # plt.plot(x_, y_)


x_test = [128.15,45.00,141.43,106.27,99.00,53.84,85.36,70.00,162.00,114.60]
pred_test = 1/(1+tf.exp(-(w*(x_test-np.mean(x))+b)))
y_test = tf.where(pred_test<0.5,0,1)

for i in range(len(x_test)):
    print(x_test[i],'\t',pred_test[i].numpy(),'\t',y_test[i].numpy())

plt.scatter(x_test,y_test)
x_ = np.array(range(-80,80))
y_ = 1 / (1 + tf.exp(-(w * x_ + b)))
plt.plot(x_+np.mean(x), y_)
# plt.scatter(x_train,y_train)

# loss = -tf.reduce_sum(y*tf.math.log(pred)+(1-y)*tf.math.log(1-pred))
# loss_mean = -tf.reduce_mean(y*tf.math.log(pred)+(1-y)*tf.math.log(1-pred))
#
# print(loss)
# print(loss_mean)
plt.show()