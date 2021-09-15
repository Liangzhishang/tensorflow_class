import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

#打开文件
img=Image.open("lena.tiff")
img1=Image.open('lena.bmp')
img2=Image.open('lena.jpg')

#img.save('lena.bmp')
#img.save('lena.jpg')
#图像格式
# print('文件格式：')
# print(img.format)
# print(img1.format)
# print(img2.format)
# print('文件尺寸：')
# print(img.size)
# print(img1.size)
# print(img2.size)
# print('文件模式：')
# print(img.mode)
# print(img1.mode)
# print(img2.mode)

#显示图像
plt.subplot(221)
plt.axis('off')
plt.imshow(img)
plt.title(img.format)

plt.subplot(222)
plt.imshow(img1)
plt.title(img1.format)

plt.subplot(223)
plt.imshow(img2)
plt.title(img2.format)

#转换图像的色彩模式
img3=img.convert('L')
print(img3.mode)
plt.subplot(224)
plt.imshow(img3)
plt.title(img3.format)

#显示图像：imshow（）

plt.show()


