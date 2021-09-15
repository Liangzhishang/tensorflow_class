import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

#加载数据集

mnist = tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y) = mnist.load_data()
# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)

#数据预处理
X_train = tf.cast(train_x,dtype= tf.float32)/255.0
X_test = tf.cast(test_x,dtype= tf.float32)/255.0
y_train = tf.cast(train_y,dtype= tf.float32)/255.0
y_test = tf.cast(test_y,dtype= tf.float32)/255.0

X_train = train_x.reshape(60000,28,28,1)
X_test = test_x.reshape(10000,28,28,1)

#建立模型
model = tf.keras.Sequential([
    #unit1
    tf.keras.layers.Conv2D(16,kernel_size=(3,3),padding="same",activation=tf.nn.relu,input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    #unit2
    tf.keras.layers.Conv2D(32,kernel_size=(3,3),padding="same",activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    #unit3
    tf.keras.layers.Flatten(),

    #unit4
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation="softmax")
])
#查看摘要
model.summary()

#配置训练方法
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])#loss='sparse_categorical_crossentropy'

#训练模型
model.fit(X_train,y_train,batch_size=64,epochs=5,validation_split=0.2)

#评估模型
model.evaluate(X_test,y_test,verbose=2)
