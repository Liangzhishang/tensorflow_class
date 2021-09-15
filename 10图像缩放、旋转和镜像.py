import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img = Image.open('lena.bmp')
#缩放
img_small = img.resize((64,64))
plt.imshow(img_small)
plt.show()
#img_small.save('lena_s.jpg')

#旋转：  图像对象.transpose(旋转方式)
"""
Image.FLIP_LEFT_RIGHT：水平翻转
Image.FLIP_TOP_BOTTOM：上下翻转
Image.ROTATE_90：逆时针旋转90°
Image.ROTATE_180：逆时针旋转180°
Image.ROTATE_270：逆时针旋转270°
Image.TRANSPOSE：将图像进行转置
Image.TRANSVERSE：将图像进行转置，再水平翻转
"""
img_r90 = img.transpose(Image.ROTATE_90)
plt.imshow(img_r90)
plt.show()

#裁剪图像：
img_region = img.crop((100,100,400,400))
plt.imshow(img_region)
plt.show()