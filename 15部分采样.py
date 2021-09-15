import tensorflow as tf

#索引
# a=tf.range(25)
#
# print(a)
# print(a[10])
# print(a[0:10])#切片


#切片：    起始位：结束位：步长
#二维张量切片
# b=tf.range(25)
# b1=tf.reshape(b,(5,5))
# print(b1)
#
# """
# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]
#  [15 16 17 18 19]
#  [20 21 22 23 24]]
# """
# #二维张量切片
# print(b1[0,0:2])#第0个样本的0到2个数据
# print(b1[2,0:2])
# print(b1[0:4,0])

#三维张量切片
# c=tf.range(24)
# c1=tf.reshape(c,(2,4,3))
# print(c1)
"""
[[[ 0  1  2]
  [ 3  4  5]
  [ 6  7  8]
  [ 9 10 11]]

 [[12 13 14]
  [15 16 17]
  [18 19 20]
  [21 22 23]]]
"""
# print(c1[1,0,0:2])

#数据提取

# d=tf.range(5)
# d1=tf.gather(d,indices=[0,2,3])
# print(d1)#[0 2 3]

#对多维张量采样

# e=tf.range(20)
# e1=tf.reshape(e,(4,5))
# # print(e1)
# """
# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]
#  [15 16 17 18 19]]
# """
# #行采样
# e2=tf.gather(e1,axis=0,indices=[0,2,3])
# print(e2)
# #列采样
# e3=tf.gather(e1,axis=1,indices=[0,2,3])
# print(e3)

# 同时采样多个点
f=tf.range(20)
f1=tf.reshape(f,(4,5))
print(f1)
"""
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
"""
f2=tf.gather_nd(f1,[[0,0],[1,1],[2,3]])
print(f2)

print(f1[0,0])
print(f1[1,1])
print(f1[2,3])