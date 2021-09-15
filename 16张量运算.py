import tensorflow as tf

"""
tf.add(x, y) 将x和y逐元素相加
tf.subtract(x, y) 将x和y逐元素相减
tf.multiply(x, y) 将x和y逐元素相乘
tf.divide(x, y) 将x和y逐元素相除
tf.math.mod(x, y) 对x逐元素取模
"""

# a1=tf.constant([0.,1.,2.])
# a2=tf.constant([3.,4.,5.])
# x=a1
# y=a2

# print(tf.add(a1,a2))# +
# print(tf.subtract(a1,a2))# -
# print(tf.multiply(a1,a2))# *
# print(tf.divide(a1,a2))
# print(tf.math.mod(a1,a2))

"""
tf.pow(x, y) 对x求y的幂次方
tf.square(x) 对x逐元素求计算平方
tf.sqrt(x) 对x逐元素开平方根
tf.exp(x) 计算e的x次方
tf.math.log(x) 计算自然对数,底数为e
"""
"""
[0   1   2]
[3   4   5]
"""
# print(tf.pow(x,y))#要同类型
# print(tf.square(x))#要浮点型
# print(tf.sqrt(y))
# print(tf.exp(x))
# print(tf.math.log(x))

#二维张量幂运算
# b1=tf.constant([[1.,4.],[9.,16.]])
# b2=tf.constant([[8.,16.],[2.,3.]])
# print(tf.pow(b1,b2))
# print(tf.pow(b1,0.5))

# #自然指数和自然对数运算
# print(tf.exp(1.))
# c=tf.exp(3.)
# print(tf.math.log(c))

"""
tf.sign(x) 返回x的符号
tf.abs(x) 对x逐元素求绝对值
tf.negative(x) 对x逐元素求相反数，y = -x
tf.reciprocal(x) 取x的倒数
tf.logical_not(x) 对x逐元素求的逻辑非
tf.ceil(x) 向上取整
tf.floor(x) 向下取整
tf.rint(x) 取最接近的整数
tf.round(x) 对x逐元素求舍入最接近的整数
tf.maximum(x, y) 返回两tensor中的最大值
tf.minimum(x, y)  返回两tensor中的最小值
"""
# x=tf.constant([-2.,-1.,1.,2.])
# y=tf.constant([-4.,-3.,3.,4.])
# print(tf.sign(x) )
# print(tf.abs(x) )
# print(tf.negative(x))
# # print(tf.(x))
# # print(tf.logical_not(x))
# # print(tf.ceil(x) )
# # print(tf.floor(x) )
# # print(tf.rint(x) )
# print(tf.round(x) )
# # print(tf.maximum(x, y))
# print(tf.minimum(x, y) )
"""
重载 运算符
"""
# print(x+y)
# print(x/y)
# print(x**y)
# print(-x)
# print(abs(x))
# print(x>y) #[ True  True False False]

#广播机制
# c=tf.constant([1,2,3])
# d=tf.constant(tf.range(12))
# d1=tf.reshape(d,(4,3))
# d2=tf.reshape(d,(2,2,3))
#
# print(d2)
#
# print(1+d2)

#张量乘法
# e=tf.range(6)
# print(e)
# e1=tf.reshape(e,(3,2))
# e2=tf.reshape(e,(2,3))
# print(e2)
# print(e1)
#
# print(e2@e1)#矩阵乘法

#多维向量乘法：三维张量*三维张量：最后两位做向量乘法，高维采用广播机制
f=tf.range(6)
f1=tf.reshape(f,(2,3))
print(f1)

print(tf.reduce_sum(f1,axis=1))
print(tf.reduce_mean(f1,axis=1))
print(tf.reduce_max(f1,axis=1))
print(tf.reduce_min(f1,axis=1))
