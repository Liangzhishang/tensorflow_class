import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

boston_housing = tf.keras.datasets.boston_housing
(train_x,train_y),(test_x,test_y) = boston_housing.load_data()

x= np.array([[3.,10.,500.],[2.,20.,200.],[1.,30.,100.],[5.,50.,100.]])

num_train = len(train_x)
num_test = len(test_x)

# print(x.min(axis=0))
# print(x.max(axis=0))
# print(x.max(axis=0)-x.min(axis=0))
for i in range(x.shape[1]):
    x[:,i]=(x[:,i]-x[:,i].min())/(x[:,i].max()-x[:,i].min())

# print(x)
x_train = (train_x-train_x.min(axis=0))/(train_x.max(axis=0)-train_x.min(axis=0))
y_train = train_y

x_test = (test_x-test_x.min(axis=0))/(test_x.max(axis=0)-test_x.min(axis=0))
y_test = test_y

x0_train = np.ones(num_train).reshape(-1,1)
x0_test = np.ones(num_test).reshape(-1,1)

X_train = tf.cast(tf.concat([x0_train,x_train],axis=1),tf.float32)
X_test = tf.cast(tf.concat([x0_test,x_test],axis=1),tf.float32)

Y_train = tf.constant(y_train.reshape(-1,1),tf.float32)
Y_test = tf.constant(y_test.reshape(-1,1),tf.float32)

learn_rate = 0.01
iter = 2000
display_step = 50

np.random.seed(612)
W = tf.Variable(np.random.randn(14,1),dtype=tf.float32)

mse_train = []
mse_test = []

for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        PRED_train = tf.matmul(X_train,W)
        Loss_train = 0.5*tf.reduce_mean(tf.square(Y_train-PRED_train))

        PRED_test = tf.matmul(X_test,W)
        Loss_test = 0.5*tf.reduce_mean(tf.square(Y_test-PRED_test))

    mse_train.append(Loss_train)
    mse_test.append(Loss_test)

    dL_dW = tape.gradient(Loss_train,W)
    W.assign_sub(learn_rate*dL_dW)

    if i%display_step ==0:
        print(f'i:{i},train loss:{Loss_train},test loss:{Loss_test}')
