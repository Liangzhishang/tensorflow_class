import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#********************加载数据****************************
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1],TRAIN_URL)

TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1],TEST_URL)

df_iris_train = pd.read_csv(train_path,header=0)
df_iris_test = pd.read_csv(test_path,header=0)

iris_train = np.array(df_iris_train)
iris_test = np.array(df_iris_test)

#********************加载数据****************************
# print(iris_test.shape)
# print(iris_train.shape)
"""
(30, 5)
(120, 5)
"""

#********************数据预处理***************************
x_train = iris_train[:,0:4]
y_train = iris_train[:,4]

x_test = iris_test[:,0:4]
y_test = iris_test[:,4]

x_train = x_train-np.mean(x_train,axis=0)
x_test = x_test-np.mean(x_test,axis=0)

X_train = tf.cast(x_train,tf.float32)
Y_train = tf.one_hot(tf.constant(y_train,dtype=tf.int32),3)

X_test = tf.cast(x_test,tf.float32)
Y_test = tf.one_hot(tf.constant(y_test,dtype=tf.int32),3)

# ********************数据预处理***************************

# print(x_train)
"""
[[ 0.555      -0.265       1.86083333  1.00333333]
 [-0.845      -0.765      -0.43916667 -0.19666667]
 [-0.945      -0.565       0.76083333  0.50333333]
 ...
  [-1.045      -0.065      -2.33916667 -1.09666667]
 [-0.345      -0.665      -0.03916667 -0.19666667]]
"""
# print(X_train)
"""
[[ 0.555      -0.265       1.8608333   1.0033333 ]
 [-0.845      -0.765      -0.43916667 -0.19666667]
 [-0.945      -0.565       0.7608333   0.50333333]
 ...
  [-1.045      -0.065      -2.3391666  -1.0966667 ]
 [-0.345      -0.665      -0.03916667 -0.19666667]]
 
"""
# print(y_train)
"""
[2. 1. 2. 0. 0. 0. 0. 2. 1. 0. 1. 1. 0. 0. 2. 1. 2. 2. 2. 0. 2. 2. 0. 2.
 2. 0. 1. 2. 1. 1. 1. 1. 1. 2. 2. 2. 2. 2. 0. 0. 2. 2. 2. 0. 0. 2. 0. 2.
 0. 2. 0. 1. 1. 0. 1. 2. 2. 2. 2. 1. 1. 2. 2. 2. 1. 2. 0. 2. 2. 0. 0. 1.
 0. 2. 2. 0. 1. 1. 1. 2. 0. 1. 1. 1. 2. 0. 1. 1. 1. 0. 2. 1. 0. 0. 2. 0.
 0. 2. 1. 0. 0. 1. 0. 1. 0. 0. 0. 0. 1. 0. 2. 1. 0. 2. 0. 1. 1. 0. 0. 1.]
"""
# print(Y_train)
"""
[[0. 0. 1.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 ...
 [1. 0. 0.]
 [0. 1. 0.]]
"""
#************************设置超参数***********************
learn_rate = 0.5
iter = 50
display_step = 10

np.random.seed(612)
W = tf.Variable(np.random.randn(4,3),dtype=tf.float32)
B = tf.Variable(np.zeros([3]),dtype=tf.float32)

#************************设置超参数***********************
# print(W)
"""
[[-0.01337706, -1.1628988 , -0.22487308],
 [ 1.1156292 ,  0.5083097 , -0.1479853 ],
 [ 0.2678837 , -0.6799997 , -0.29333967],
 [-0.38372967, -1.2842919 ,  1.4081773 ]]
"""
# print(B)
"""
[0., 0., 0.]
"""
#***********************训练模型**************************
acc_train = []
acc_test = []
cce_train = []
cce_test = []

for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        PRED_train = tf.nn.softmax(tf.matmul(X_train,W)+B)#先矩阵相乘，再归一化
        Loss_train = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=Y_train,y_pred=PRED_train))#交叉熵损失函数

    PRED_test = tf.nn.softmax(tf.matmul(X_test, W) + B)  # 先矩阵相乘，再归一化
    Loss_test = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=Y_test, y_pred=PRED_test))  # 交叉熵损失函数

    accuracy_train = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PRED_train.numpy(),axis=1),y_train),tf.float32))#准确率
    accuracy_test = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PRED_test.numpy(), axis=1), y_test), tf.float32))

    acc_train.append(accuracy_train)
    acc_test.append(accuracy_test)
    cce_train.append(Loss_train)
    cce_test.append(Loss_test)

    grads = tape.gradient(Loss_train,[W,B])#求导
    W.assign_sub(learn_rate*grads[0])
    B.assign_sub(learn_rate*grads[1])

    if i%display_step==0:
        print(f'i:{i},train_acc:{accuracy_train},train_lost:{Loss_train},test_acc:{accuracy_test},test_loss:{Loss_test}')
        print(grads)

#***********************训练模型**************************
"""
i:0,train_acc:0.3333333432674408,train_lost:2.0669775009155273,test_acc:0.2666666805744171,test_loss:1.8808554410934448
i:10,train_acc:0.875,train_lost:0.3394101560115814,test_acc:0.8666666746139526,test_loss:0.46170514822006226
i:20,train_acc:0.875,train_lost:0.27964702248573303,test_acc:0.8666666746139526,test_loss:0.3684142231941223
i:30,train_acc:0.9166666865348816,train_lost:0.24592377245426178,test_acc:0.9333333373069763,test_loss:0.31481361389160156
i:40,train_acc:0.9333333373069763,train_lost:0.2229219377040863,test_acc:0.9333333373069763,test_loss:0.2786431610584259
i:50,train_acc:0.9333333373069763,train_lost:0.20563557744026184,test_acc:0.9666666388511658,test_loss:0.2519374489784241

grads:
[[ 0.0039493 , -0.01062285,  0.00667354],
 [-0.00785747,  0.00568526,  0.00217221],
 [ 0.02159379,  0.00499068, -0.02658447],
 [ 0.00887238,  0.00891174, -0.01778411]]
"""
#*********************结果可视化***************************
plt.figure(figsize=(10,3))

plt.subplot(121)
plt.plot(cce_train,color='blue',label='train')
plt.plot(cce_test,color='red',label='test')
plt.xlabel("iteration")
plt.ylabel("loss")

plt.subplot(122)
plt.plot(acc_train,color='blue',label='train')
plt.plot(acc_test,color='red',label='test')
plt.xlabel("iteration")
plt.ylabel("accuracy")

plt.show()
#*********************结果可视化***************************
