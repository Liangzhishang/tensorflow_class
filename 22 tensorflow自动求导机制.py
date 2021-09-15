"""

自动求导机制
Variable对象

"""

import tensorflow as tf

import numpy as np

# print(tf.Variable(3.))
# print(tf.Variable([1,2]))
# print(tf.Variable(np.array([1,2])))
# a=tf.Variable(3.)
# print(a)

#将张量封装成可训练变量
# b=tf.Variable(tf.constant([[1,2],[3,4]]))
# print(b)
#
# c=tf.Variable([[1,2],[3,4]])
# print(c)
#
# print(c.trainable)
# print(b.trainable)

#可训练变量赋值
# d=tf.Variable([1,2])
# d.assign([3,4])#赋值
# # print(d.assign([3,4]))
# print(d)
#
# d.assign_add([1,1])#加法
# print(d)
# d.assign_add([1,1])
# print(d)
# d.assign_sub([1,1])

#isinstance()方法

e=tf.range(5)
f=tf.Variable(e)

print(e)
print(f)

print(isinstance(e,tf.Tensor),isinstance(e,tf.Variable))
print(isinstance(f,tf.Tensor),isinstance(f,tf.Variable))
