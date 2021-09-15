import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

#灰度图像转化成数组：
img_gary = Image.open('lena_gory.bmp')
arr_img_gray = np.array(img_gary)

# print(arr_img_gray.shape)
# print(arr_img_gray)

#彩色图像转化成数组：
img = Image.open('lena.bmp')
arr_img = np.array(img)

print(arr_img.shape)
print(arr_img)
