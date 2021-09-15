import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# mnist = tf.keras.datasets.mnist
# (train_x,train_y),(text_x,text_y) = mnist.load_data

mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data("F:\研究生学习\神经网络与深度学习——tensorflow2.0实战\mnist.npz")

# print(f'train_x:{len(train_x)}')
# print(f'text_x:{len(test_x)}')

# print(train_x[0])

# plt.axis('off')
# plt.imshow(train_x[0],cmap="gray")

# print(train_y[0])
# plt.show()
for i in range(4):
    num = np.random.randint(1,50000)

    plt.subplot(1,4,i+1)
    plt.axis('off')
    plt.imshow(train_x[num],cmap='gray')
    plt.title(train_y[num])

plt.show()
