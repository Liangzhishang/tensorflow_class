import tensorflow as tf
import numpy as np

# b = np.arange(12)
# print(b)
#
# #改变数组形状：
#
# c = b.reshape(3,4)
# print(c)
# print()
#
# d=np.arange(12).reshape(3,4)
# print(d)
# print()
#
# e=b.reshape(-1,2)
# print(e)

#数组运算
# f= np.array([0,1,2,3])
# g= np.array([4,5,6,7])
# # print(f+g)
# # print(f**2)#幂运算
# # print(f*2)#乘法
# # print(f*g)#矩阵相乘
#
# print(np.transpose(f))#转置
# # print(np.linalg.inv(f))#求逆
#
# print(np.sum(f))
#
# g=np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11]])
# print(np.sum(g,axis=1))

h=np.array([1,2,3])
i=np.array([4,5,6])

print(np.stack((h,i),axis=0))
print(np.stack((h,i),axis=1))

