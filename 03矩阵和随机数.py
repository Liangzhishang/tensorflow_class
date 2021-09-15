import numpy as np

#矩阵
# print(np.mat('1 2 3 ; 4 5 6'))
#
# print(np.mat([[1,2,3],[4,5,6]]))
# a=np.mat('1 2 3 ; 4 5 6')
# b=np.array([[1,2,3],[4,5,6]])
# print(np.ndim(a))#矩阵的维数
# print(np.shape(a))#矩阵的形状
# print(np.size(a))#矩阵的元素个数
# print(b)#矩阵的数据类型
#

#矩阵运算

# a1=np.mat([[0,1],[2,3]])
# a2=np.mat([[1,1],[2,0]])
# a3=a1*a2#矩阵相乘
# print(a3)

# n=np.mat([[1,2],[-1,-3]])
# print(n)
# print()
#
# print(n.T)#转置
# print()
#
# print(n.I)#求逆
# print()
#
# print(n*n.I)
#
# a=np.array([[1,2,3],[4,5,6]])
# m=np.mat(a)
# print(m)
# print(m.I)
# print(m.T)

# print(np.random.rand(2,3))#[0,1)分布的随机数组，结构是2*3
# print()
#
# print(np.random.uniform(5,10,(2,3)))
# print()
#
# print(np.random.randint(1,5,(3,2)))
# print()
#
# print(np.random.randn(3,3))

arr = np.arange(10)
print(arr)
np.random.shuffle(arr)
print(arr)
