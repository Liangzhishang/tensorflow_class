import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"

train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1],TRAIN_URL)

df_iris = pd.read_csv(train_path,header=0)

iris=np.array(df_iris)#数据集
# print(iris)
# print(iris.shape)#(120, 5)
#花萼长度   花萼宽度    花瓣长度    花萼宽度
train_x = iris[:,0:2]#前两个属性
train_y = iris[:,4]#鸢尾花种类
# print(train_x)
# print(train_y)
# print(train_x.shape,train_y.shape)

x_train = train_x[train_y<2]#前两个属性
y_train = train_y[train_y<2]#鸢尾花种类
# print(x_train)
# print(y_train)
# print(x_train.shape,y_train.shape)
num = len(x_train)

cm_pt = mpl.colors.ListedColormap(["blue","red"])
# plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=cm_pt)

#属性中心化
x_train=x_train-np.mean(x_train,axis=0)#转化成平均值是0的
plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=cm_pt)
# plt.show()

#生成多元模型的属性矩阵和标签列向量
x0_train=np.ones(num).reshape(-1,1)
X = tf.cast(tf.concat((x0_train,x_train),axis=1),tf.float32)
Y = tf.cast(y_train.reshape(-1,1),tf.float32)
# print(x_train)
# print(X)
# print(Y)
#设置超参数
learn_rate = 0.2
iter = 120
display_step = 30

#设置模型参数初始值
np.random.seed(612)
W = tf.Variable(np.random.randn(3,1),dtype=tf.float32)
# print(W)
ce=[]
acc = []
pred = []
for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        PRED = 1/(1+tf.exp(-tf.matmul(X,W)))
        Loss = -tf.reduce_mean(Y*tf.math.log(PRED)+(1-Y)*tf.math.log(1-PRED))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.where(PRED.numpy()<0.5,0.,1.),Y),tf.float32))

    ce.append(Loss)
    acc.append(accuracy)
    pred.append(PRED[0])

    #求导
    dL_dW = tape.gradient(Loss,W)
    # print(f'dl_dw:{dL_dW}')
    #迭代
    W.assign_sub(learn_rate*dL_dW)

    if i %display_step==0:
        print(f'i:{i},acc:{accuracy},loss:{Loss}')
# print(pred)
# plt.figure(figsize=(5,3))
x_ = [-1.5,1.5]
y_ = -(W[0]+W[1]*x_)/W[2]
plt.plot(x_,y_,'g')

# plt.plot(ce,color ='blue',label='loss')
# plt.plot(acc,color='red',label='acc')
# plt.plot(pred,color='green',label='pred')
plt.legend()
plt.show()

