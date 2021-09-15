import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#******************加载数据**************************
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1],TRAIN_URL)

TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1],TEST_URL)

df_iris_train = pd.read_csv(train_path,header=0)
df_iris_test = pd.read_csv(test_path,header=0)

iris_train = np.array(df_iris_train)
iris_test = np.array(df_iris_test)

#******************加载数据**************************
# print(iris_train.shape)
# print(iris_test.shape)

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
#********************数据预处理***************************

# print(X_train.shape)
# print(Y_train.shape)


#****************建立模型***************************

#建立Sequential模型
model = tf.keras.Sequential()
#添加层

#输入数据的形状，不进行计算，只是进行形状的转换
model.add(tf.keras.layers.Flatten(input_shape=(4,)))#输入层，4
#                        输入该网络层的数据，激活函数
model.add(tf.keras.layers.Dense(16,activation='relu'))#隐含层，16
model.add(tf.keras.layers.Dense(3,activation='softmax'))#输出层，3

model.summary()#查看摘要
#****************建立模型***************************

#************配置训练方法****************************
#           optimizer:优化器   loss:损失函数   metrics:性能评估函数
model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
#************配置训练方法****************************

#************训练模型*******************************
#   训练集的输入特征，训练集的标签，批量大小，迭代次数，从训练集划分多少比例给测试集
model.fit(X_train,y_train,batch_size=1,epochs=25,validation_split=0)

#************训练模型*******************************

#************评估模型*******************************
model.evaluate(X_test,y_test,verbose=2)

#************评估模型*******************************

#************使用模型*******************************
# print(model.predict([[X_test[0]]]))
# plt.axis("off")
# plt.imshow(test_x[0],cmap='gray')
# plt.show()
# print(y_test[0])
# y_pred=np.argmax(model.predict([[X_test[0]]]))
# print(y_pred)
while 1:
    test_pred = int(input("input the num:"))
    print(np.argmax(model.predict([X_test[test_pred:test_pred + 1]]), axis=1))
    print()
test_pred = 1
#              使用模型
print(np.argmax(model.predict([X_test[test_pred:test_pred+1]]),axis=1))
#************使用模型*******************************