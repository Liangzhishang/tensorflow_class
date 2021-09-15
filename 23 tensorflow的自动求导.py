"""
with GradientTape() as tape:
函数表达式
grad=tape.gradient(函数,自变量)

"""

import tensorflow as tf

import numpy as np

# x = tf.Variable(3.)
# with tf.GradientTape(persistent=True) as tape:
#     y = tf.square(x)
#     z = pow(x,3)
#
# dy_dx = tape.gradient(y,x)#grad=tape.gradient(函数,自变量)
# dz_dx = tape.gradient(z,x)
# print(y)
# print(dy_dx)
#
# print(z)
# print(dz_dx)
# del tape

#多元函数求偏导数
# x = tf.Variable(3.)
# y = tf.Variable(4.)
#
# with tf.GradientTape(persistent=True) as tape:
#     f = tf.square(x) + 2*tf.square(y)+1
#
# # df_dx,df_dy = tape.gradient(f,[x,y])
# # first_grads = tape.gradient(f,[x,y])
# df_dx = tape.gradient(f,x)
# df_dy = tape.gradient(f,y)
#
# print(f)
# print(df_dx)
# print(df_dy)
# # print(first_grads)
#
# del tape

#求二阶导数

# x = tf.Variable(3.)
# y = tf.Variable(4.)
# with tf.GradientTape() as tape2:
#
#     with tf.GradientTape() as tape1:
#         f=tf.square(x) + 2*tf.square(y) + 1
#     first_grads = tape1.gradient(f,[x,y])
#
# second_grads = tape2.gradient(first_grads,[x,y])
#
# print(f)
# print(first_grads)#一阶导数
# print(second_grads)#二阶导数
#
# del tape2
# del tape1

#对向量求偏导数
x = tf.Variable([1.,2.,3.])#必须浮点型
y = tf.Variable([4.,5.,6.])

with tf.GradientTape(persistent=True) as tape:
    f = tf.square(x) + 2*tf.square(y)+1

# df_dx,df_dy = tape.gradient(f,[x,y])
#
# print(f)
# print(df_dx)
# print(df_dy)

with tf.GradientTape(persistent=True) as tape2:

    with tf.GradientTape(persistent=True) as tape1:
        f=tf.square(x) + 2*tf.square(y) + 1
    first_grads1,first_grads2 = tape1.gradient(f,[x,y])

print(first_grads1,first_grads2)

del tape2
del tape1