import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
#*******************************************************加载数据集******************************************************************
#加载数据集
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1],TRAIN_URL)
df_iris_train = pd.read_csv(train_path,header=0)
#*******************************************************加载数据集******************************************************************

# print(train_path)
# print(df_iris_train)
"""
C:\ Users\lzs\.keras\datasets\iris_training.csv
     120    4  setosa  versicolor  virginica
0    6.4  2.8     5.6         2.2          2
1    5.0  2.3     3.3         1.0          1
2    4.9  2.5     4.5         1.7          2
3    4.9  3.1     1.5         0.1          0
4    5.7  3.8     1.7         0.3          0
..   ...  ...     ...         ...        ...
115  5.5  2.6     4.4         1.2          1
116  5.7  3.0     4.2         1.2          1
117  4.4  2.9     1.4         0.2          0
118  4.8  3.0     1.4         0.1          0
119  5.5  2.4     3.7         1.0          1

[120 rows x 5 columns]
"""
#*************************************处理数据************************************
#处理数据
iris_train = np.array(df_iris_train)#
x_train = iris_train[:,2:4]
y_train = iris_train[:,4]
num_train = len(x_train)

x0_train = np.ones(num_train).reshape(-1,1)
X_train = tf.cast(tf.concat([x0_train,x_train],axis=1),tf.float32)
Y_train = tf.one_hot(tf.constant(y_train,dtype=tf.int32),3)
#*************************************处理数据*************************************
# print(X_train)
"""
[[1.  5.6 2.2]
 [1.  3.3 1. ]
 [1.  4.5 1.7]
 ....
 [1.  1.4 0.1]
 [1.  3.7 1. ]]
 shape=(120, 3)
"""
#*******************************设置超参数，初始值***********************************
learn_rate = 0.2
iter = 500
display_step = 100
np.random.seed(612)
W = tf.Variable(np.random.randn(3,3),dtype=tf.float32)
#*******************************设置超参数，初始值***********************************
# print(W)
"""
[[-0.01337706, -1.1628988 , -0.22487308],
 [ 1.1156292 ,  0.5083097 , -0.1479853 ],
 [ 0.2678837 , -0.6799997 , -0.29333967]]
"""
#******************************训练模型*********************************************
acc = []
cce = []

for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        PRED_train = tf.nn.softmax(tf.matmul(X_train,W))#矩阵相乘，softmax():归一化
        Lost_train = -tf.reduce_sum(Y_train*tf.math.log(PRED_train))/num_train  #平均交叉熵损失

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PRED_train.numpy(),axis=1),y_train),tf.float32))#准确率

    acc.append(accuracy)
    cce.append(Lost_train)

    dL_dW = tape.gradient(Lost_train,W)
    W.assign_sub(learn_rate*dL_dW)

    # if i%display_step==0:
    #     print(f'i:{i},acc准确率:{accuracy},loss平均交叉熵损失:{Lost_train}')
        # print(W)



#******************************训练模型*********************************************

#******************************训练结果*********************************************
# print('结束')
# print(W)
"""
[[ 4.0687   , -1.0461913, -4.423658 ],
 [-0.5528681,  1.2314469,  0.7973755],
 [-1.7617267, -1.2250443,  2.2813148]]
"""
# print(PRED_train)
# print(tf.reduce_sum(PRED_train,axis=1))
# # print(PRED_train)
#
# #转换成自然顺序码
# print(tf.argmax(PRED_train.numpy(),axis=1))#最大值的索引
#*****************************训练结果**********************************************

#*****************************绘制分类图*************
M = 500
x1_min,x2_min = x_train.min(axis=0)
x1_max,x2_max = x_train.max(axis=0)
t1=np.linspace(x1_min,x1_max,M)#在指定的间隔内返回均匀间隔的数字
t2=np.linspace(x2_min,x2_max,M)
m1,m2 = np.meshgrid(t1,t2)#生成坐标矩阵

m0 = np.ones(M*M)
X_ = tf.cast(np.stack((m0,m1.reshape(-1),m2.reshape(-1)),axis=1),tf.float32)
Y_ = tf.nn.softmax(tf.matmul(X_,W))#归一化
# print(Y_)

Y_ = tf.argmax(Y_.numpy(),axis=1)#最大值的索引值

n= tf.reshape(Y_,m1.shape)
plt.figure(figsize=(8,6))
cm_bg = mpl.colors.ListedColormap(["white","pink","blue"])

plt.pcolormesh(m1,m2,n,cmap=cm_bg)
plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap="brg")#散点图

plt.show()

#*****************************绘制分类图*************
"""
[9.98490810e-01 1.31047703e-03 1.98655413e-04]
 [9.76865411e-01 1.61640421e-02 6.97060442e-03]
 [9.94857192e-01 4.09078924e-03 1.05206843e-03]
 [8.12450171e-01 9.41383392e-02 9.34114978e-02]
 [8.58580053e-01 7.28899017e-02 6.85300380e-02]
 ....
  [8.09499502e-01 9.06558558e-02 9.98446271e-02]
 [7.97711492e-01 9.82180238e-02 1.04070485e-01]
 [9.83010888e-01 1.27576739e-02 4.23137611e-03]]
 shape=(120, 3)
"""
# a=[0,2,3,5]
# b=tf.one_hot(a,6)
# # print(b)
"""
[[1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 1.]]
"""

# pred = np.array([[0.1,0.2,0.7],[0.1,0.7,0.2],[0.3,0.4,0.3]])
# y = np.array([2,1,0])
# y_onehot = np.array([[0,0,1],[0,1,0],[1,0,0]])

# print(tf.argmax(pred,axis=1))#最大值的索引
# print(tf.equal(tf.argmax(pred,axis=1),y))
# print(tf.cast(tf.equal(tf.argmax(pred,axis=1),y),tf.float32))
# print(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,axis=1),y),tf.float32)))#准确率
"""
tf.Tensor([2 1 1], shape=(3,), dtype=int64)
tf.Tensor([ True  True False], shape=(3,), dtype=bool)
tf.Tensor([1. 1. 0.], shape=(3,), dtype=float32)
tf.Tensor(0.6666667, shape=(), dtype=float32)
"""

# print(-y_onehot*tf.math.log(pred))
# print(-tf.reduce_sum(y_onehot*tf.math.log(pred)))#所有样本交叉熵之和
# print(-tf.reduce_sum(y_onehot*tf.math.log(pred))/len(pred))#平均交叉熵损失

"""
tf.Tensor(
[[-0.         -0.          0.35667494]样本1
 [-0.          0.35667494 -0.        ]样本2
 [ 1.2039728  -0.         -0.        ]样本3], shape=(3, 3), dtype=float64)
tf.Tensor(1.917322692203401, shape=(), dtype=float64)
tf.Tensor(0.6391075640678003, shape=(), dtype=float64)
"""




