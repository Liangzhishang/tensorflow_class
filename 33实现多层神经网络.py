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

#************************设置超参数***********************
learn_rate = 0.5
iter = 50
display_step = 10

np.random.seed(612)
W1 = tf.Variable(np.random.randn(4,16),dtype=tf.float32)#隐含层4个输入，16个节点
B1 = tf.Variable(np.zeros([16]),dtype=tf.float32)

W2 = tf.Variable(np.random.randn(16,3),dtype=tf.float32)#输出层16个输入，3个节点
B2 = tf.Variable(np.zeros([3]),dtype=tf.float32)
#************************设置超参数***********************

# print(W1)
'''
[[-0.01337706, -1.1628988 , -0.22487308,  1.1156292 ,  0.5083097 ,
        -0.1479853 ,  0.2678837 , -0.6799997 , -0.29333967, -0.38372967,
        -1.2842919 ,  1.4081773 ,  1.0663726 ,  0.9593439 , -0.9004021 ,
         2.0515325 ],
       [-0.10351721, -0.16258761,  0.45697057,  0.8329825 ,  0.45354867,
         1.1457878 , -0.7355144 ,  1.0014898 , -0.25348657,  0.68232054,
        -1.1784879 , -1.8347055 ,  1.3391234 ,  0.7349712 ,  0.84466636,
         0.6133882 ],
       [-0.14146568,  0.27880755,  0.41646856, -0.32077375, -0.84231806,
         0.5726403 ,  0.6158719 ,  1.0001477 ,  0.08818302, -2.0345726 ,
        -1.6595993 , -0.00649302, -0.59628373,  0.07983295,  0.03992981,
         0.07073665],
       [-1.333659  , -0.45797968, -2.0257986 ,  0.20677613,  0.44964716,
        -0.7598014 , -0.5111772 ,  0.24023303,  0.33921796,  0.65154195,
         1.0760499 ,  0.79061824, -0.14383858,  0.8689316 ,  0.20395447,
         0.2681641 ]]
'''
# print(B1)
'''
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
'''
#***********************训练模型***********************
acc_train = []
acc_test = []
cce_train = []
cce_test = []

for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        Hidden_train = tf.nn.relu(tf.matmul(X_train,W1)+B1)#隐含层
        PRED_train = tf.nn.softmax(tf.matmul(Hidden_train,W2)+B2)#输出层
        Loss_train = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=Y_train,y_pred=PRED_train))

        Hidden_test = tf.nn.relu(tf.matmul(X_test, W1) + B1)  # 隐含层
        PRED_test = tf.nn.softmax(tf.matmul(Hidden_test, W2) + B2)  # 输出层
        Loss_test = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=Y_test, y_pred=PRED_test))

    accuracy_train = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PRED_train.numpy(),axis=1),y_train),tf.float32))#准确率
    accuracy_test = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PRED_test.numpy(),axis=1),y_test),tf.float32))

    acc_train.append(accuracy_train)
    acc_test.append(accuracy_test)
    cce_train.append(Loss_train)
    cce_test.append(Loss_test)

    grads = tape.gradient(Loss_train,[W1,B1,W2,B2])
    W1.assign_sub(learn_rate*grads[0])
    B1.assign_sub(learn_rate*grads[1])
    W2.assign_sub(learn_rate*grads[2])
    B2.assign_sub(learn_rate*grads[3])

    if i%display_step == 0:
        print(f'i:{i},train_acc:{accuracy_train},train_lost:{Loss_train},TEST_ACC:{accuracy_test},test_loss:{Loss_test}')
#***********************训练模型***********************

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

