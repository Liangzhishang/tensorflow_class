import matplotlib.pyplot as plt
import numpy as np

# plt.scatter(5,5,36,'b','.','tuli')
#
# plt.show()
plt.rcParams['font.sans-serif']="SimHei"#设置字数为黑体
plt.rcParams['axes.unicode_minus']=False

#生成标准正态分布散点图
n = 500
x = np.random.normal(0,1,n)
y = np.random.normal(0,1,n)

plt.scatter(x,y)#绘制数据点

plt.title('标准正态分布',fontsize=20)#设置标题
plt.text(2.5, 2.5, "均值：0\n标准差：1")#设置文本

plt.xlim(-4,4)#x轴范围
plt.ylim(-4,4)#y轴范围

plt.xlabel('横坐标x',fontsize=14)#设置x轴标签文本
plt.ylabel('纵坐标',fontsize=14)#设置y轴标签文本

plt.show()