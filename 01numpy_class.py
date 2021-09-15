import tensorflow as tf
import numpy as np

#创建数组

# a=np.array([0,1,2,3,4,5,6])
# print(a)
# print(a[0])
# print(type(a))
# print()
#
# print(a[1:3])#切片
# print(a[:3])
# print(a[0:])
# print()
#
# print(a.ndim)#数组的维数
# print(a.shape)#数组的形状
# print(a.size)#数组元素总个数
# print(a.dtype)#数组中元素的数据类型
# print(a.itemsize)#数组中每个元素的字节数

#二维数组
# b=np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11]])
#
# print(b)
# print(b[0])
# print(type(b))
# print()
#
# print(b[1:3])#切片
# print(b[:3])
# print(b[0:])
# print()
#
# print(b.ndim)#数组的维数
# print(b.shape)#数组的形状
# print(b.size)#数组元素总个数
# print(b.dtype)#数组中元素的数据类型
# print(b.itemsize)#数组中每个元素的字节数

#创建特殊的数组：
c=np.arange(0,10)#创建由数字序列构成的数组
print(c)
print()

d=np.ones((3,2),dtype=np.int16)#创建全1数组
print(d)
print()

e=np.zeros((2,3))#创建全部为0的数组
print(e)
print()

f=np.eye(3)#创建单位矩阵
print(f)
print()

print(np.linspace(1,9,5))#等差数列


