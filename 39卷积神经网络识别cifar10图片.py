import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers,Sequential
#加载数据集
cifar10 = tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)