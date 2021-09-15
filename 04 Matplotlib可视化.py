import matplotlib.pyplot as plt

#创建画布

plt.figure(figsize=(4,2),facecolor="white")
#fig=plt.figure()
#plt.plot()#绘制空白图形
#plt.show()#显示绘图

#subplot()划分子图
plt.subplot(221)
plt.title('子标题1')
plt.subplot(222)
plt.title('子标题2')
plt.subplot(223)
plt.title('子标题3')
plt.subplot(224)
plt.title('子标题4')

#设置中文字体:中文黑体
plt.rcParams["font.sans-serif"]="SimHei"

#全局标题
plt.suptitle('全局标题')

plt.show()