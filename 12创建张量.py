import tensorflow as tf
import numpy as np
# print(tf.__version__)   #tensorflow的版本
# print(tf.executing_eagerly())

#创建张量
# a=tf.constant([[1,2],[3,4]])
# print(a.numpy())
# print(type(a))
# print(a)
# print()
#
# b=tf.constant(1.0)
# c=tf.constant(1.)
# print(b)
# print(c)

# d=tf.constant(1.0,dtype=tf.float64)
# print(d)

# e=tf.constant(np.array([1,2]))
# print(e)

#数据类型转换
# f=tf.constant(np.array([1,2]))
# g=tf.cast(f,dtype=tf.float32)#类型转换的函数
# print(g)

#参数为布尔型
# h=tf.constant(True)
# print(h)

# i=tf.constant([True,False])
# j=tf.cast(i,dtype=tf.int32)
# print(i)
# print(j)
#
# k=tf.cast(j,tf.bool)
# print(k)

#参数为字符串
# l=tf.constant('hello')
# print(l)

# m=tf.constant(np.arange(12).reshape(3,4))
m=np.arange(12).reshape(3,4)
n=tf.convert_to_tensor(m)
o=tf.constant(m)
print(m)
print(n)
print(o)

#判断数据对象是否是张量
print(tf.is_tensor(m))
print(tf.is_tensor(n))
print(tf.is_tensor(o))




