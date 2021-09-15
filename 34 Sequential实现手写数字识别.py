import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#******************加载数据**************************
mnist = tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y)=mnist.load_data()

#******************加载数据**************************
# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)
"""
(60000, 28, 28)
(60000,)
(10000, 28, 28)
(10000,)

"""
#*****************数据预处理*************************
X_train = train_x.reshape((60000,28*28))
X_test = test_x.reshape((10000,28*28))

X_train,X_test = tf.cast(train_x/255.0,tf.float32),tf.cast(test_x/255.0,tf.float32)
y_train,y_test = tf.cast(train_y,tf.int16),tf.cast(test_y,tf.int16)

#*****************数据预处理*************************
# print(train_x)
# print(X_train)
# print(X_test)
# print(train_y)
# print(y_train)
"""

"""
#****************建立模型***************************

#建立Sequential模型
model = tf.keras.Sequential()
#添加层

#输入数据的形状，不进行计算，只是进行形状的转换
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))#输入层，784
#                        输入该网络层的数据，激活函数
model.add(tf.keras.layers.Dense(128,activation='relu'))#隐含层，128
model.add(tf.keras.layers.Dense(10,activation='softmax'))#输出层，10

model.summary()#查看摘要
#****************建立模型***************************
print()
"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 128)               100480    
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
"""
#************配置训练方法****************************
#           optimizer:优化器   loss:损失函数   metrics:性能评估函数
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
#************配置训练方法****************************

#************训练模型*******************************
#   训练集的输入特征，训练集的标签，批量大小，迭代次数，从训练集划分多少比例给测试集
model.fit(X_train,y_train,batch_size=64,epochs=5,validation_split=0.2)

#************训练模型*******************************
"""
Epoch 1/5
750/750 [==============================] - 2s 2ms/step - loss: 0.3443 - sparse_categorical_accuracy: 0.9036 - val_loss: 0.1855 - val_sparse_categorical_accuracy: 0.9468
Epoch 2/5
750/750 [==============================] - 2s 2ms/step - loss: 0.1573 - sparse_categorical_accuracy: 0.9547 - val_loss: 0.1361 - val_sparse_categorical_accuracy: 0.9613
Epoch 3/5
750/750 [==============================] - 2s 2ms/step - loss: 0.1113 - sparse_categorical_accuracy: 0.9677 - val_loss: 0.1149 - val_sparse_categorical_accuracy: 0.9656
Epoch 4/5
750/750 [==============================] - 2s 2ms/step - loss: 0.0839 - sparse_categorical_accuracy: 0.9756 - val_loss: 0.1092 - val_sparse_categorical_accuracy: 0.9667
Epoch 5/5
750/750 [==============================] - 2s 2ms/step - loss: 0.0669 - sparse_categorical_accuracy: 0.9808 - val_loss: 0.1011 - val_sparse_categorical_accuracy: 0.9696

"""

#************评估模型*******************************
model.evaluate(X_test,y_test,verbose=2)

#************评估模型*******************************
"""
313/313 - 0s - loss: 0.0908 - sparse_categorical_accuracy: 0.9728
"""
#************模型保存*******************************
# model.save("minist_model.h5")
#************模型保存*******************************

#************使用模型*******************************
# print(model.predict([[X_test[0]]]))
plt.axis("off")
plt.imshow(test_x[0],cmap='gray')
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