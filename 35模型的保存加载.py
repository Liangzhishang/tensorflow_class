import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#******************加载数据**************************
mnist = tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y)=mnist.load_data()

#******************加载数据**************************


#*****************数据预处理*************************
X_train = train_x.reshape((60000,28*28))
X_test = test_x.reshape((10000,28*28))

X_train,X_test = tf.cast(train_x/255.0,tf.float32),tf.cast(test_x/255.0,tf.float32)
y_train,y_test = tf.cast(train_y,tf.int16),tf.cast(test_y,tf.int16)

#*****************数据预处理*************************

#*****************加载模型***************************
model = tf.keras.models.load_model('minist_model.h5')
model.summary()
#*****************加载模型***************************

#*****************模型评估***************************
model.evaluate(X_test,y_test,verbose=2)
#*****************模型评估***************************

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
