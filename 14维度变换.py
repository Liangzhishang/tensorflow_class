import tensorflow as tf
import numpy as np
# a=tf.range(24)
# print(a)
# b=tf.reshape(a,(2,3,4))#改变张量的形状
# print(b)

# c=tf.constant(np.arange(24).reshape(2,3,4))
# print(c)

#增加维度
# d=tf.constant([1,2,3,4])
# print(d)
#
# d1=tf.expand_dims(d,0)
# print(d1)

# e=tf.range(24)
# print(e)
# e1= tf.reshape(e,(2,4,3))#shape=(2, 4, 3)
# print(e1)
#
# e2= tf.expand_dims(e1,1)#shape=(2, 1, 4, 3)
# print(e2)
#
# #删除维度
# e3 = tf.squeeze(e2,1)#shape=(2, 4, 3)
# print(e3)

#交换维度
# f=tf.constant([[1,2,3],[4,5,6]])
# print(f)
#
# g=tf.transpose(f,perm=[0,1])
# print(g)
# print(f)

# h=tf.range(24)
# i=tf.reshape(h,[2,3,4])
#
# print(h)
# print(i)
#
# j=tf.transpose(i,(2,1,0))
# print(j)#shape=(4,3,2)

#拼接张量
# k1=tf.constant([[1,2,3],[4,5,6]])
# k2=tf.constant([[7,8,9],[10,11,12]])
#
# print(k1)#shape=(2,3)
# print(k2)#shape=(2,3)
#
# k3=tf.concat((k1,k2),0)#张量拼接
# k4=tf.concat((k1,k2),1)
#
# print(k3)#shape=(4, 3)
# print(k4)#shape=(2, 6)

#分割张量
# l=tf.range(24)
# l1=tf.reshape(l,[4,6])
# print(l1)
#
# l2=tf.split(l1,(1,2,3),1)
# print(l2)

#堆叠张量
m1=tf.constant([1,2,3])
m2=tf.constant([4,5,6])
print(m1)
print(m2)

m3=tf.stack((m1,m2),axis=0)
m4=tf.stack((m2,m1),axis=0)
print(m3)
print(m4)

m5=tf.stack((m1,m2),axis=1)
m6=tf.stack((m2,m1),axis=1)
print(m5)
print(m6)

