import matplotlib.pyplot as plt
import numpy as np

n=100
# y1 = np.arange(0,10)
# y2 = [9,8,7,6,5,4,3,2,1,0]
# print(y1)

# y1=np.random.randint(27,37,n)
# y2=np.random.randint(40,60,n)
#
# plt.plot(y1,label='wendu')#生成折线图
# plt.plot(y2,label='shidu')#生成折线图

y1=[32,25,16,30,24,45,40,33,28,17,24,20]
y2=[23,-35,-26,-35,-45,-43,-35,-32,-23,-17,-22,-28]
#生成柱状图
plt.bar(range(len(y1)), y1,width=0.8,facecolor='green',edgecolor='white',label='统计量1')
plt.bar(range(len(y2)), y2,width=0.5,facecolor='red',edgecolor='white',label='统计量2')

plt.show()