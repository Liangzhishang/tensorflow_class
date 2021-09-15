import tensorflow as tf
import numpy as np

#创建全0张量和全1张量
# a=tf.zeros((2,3),dtype=tf.float32)
# b=tf.ones((2,3),dtype=tf.float32)
# c=tf.ones((6,),dtype=tf.float32)
# print(a)
# print(b)
# print(c)

#创建元素值都相同的张量
# d=tf.constant(9,shape=(2,3))
# print(d)
# e=tf.fill(dims=(2,3),value=9)
# print(e)

#创建随机数张量
#正态分布
# f=tf.random.normal((2,2))
# print(f)

# g=tf.random.normal((3,3,3),mean=0.0,stddev=2.0)#mean:均值     stddev:标准差
# print(g)

#设置随机数种子：
# tf.random.set_seed(8)
# h=tf.random.normal((2,2))
#
# tf.random.set_seed(8)
# i=tf.random.normal((2,2))
#
# print(h)
# print(i)

#创建均匀分布张量
# j=tf.random.uniform((3,3),0,10,dtype='int32')
# print(j)

#随机打乱
# k=tf.constant([[1,2],[3,4],[5,6]])
# l=tf.random.shuffle(k)#随机打乱第一维并返回，不会影响原来的
# print(k)
# print(l)

#创建序列
m=tf.range(10)
n=tf.range(10,delta=2)
o=tf.range(1,10,delta=2)
print(m)
print(n)
print(o)