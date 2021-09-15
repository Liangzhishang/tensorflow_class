import matplotlib.pyplot as plt
from PIL import Image

img=Image.open("lena.tiff")
#颜色通道的分离
img_r,img_g,img_b = img.split()

plt.figure(figsize=(10,10))
#分别显示三个通道
plt.subplot(221)
plt.axis('off')
plt.imshow(img_r,cmap='gray')
plt.title('R')

plt.subplot(222)
plt.axis('off')
plt.imshow(img_g,cmap='gray')
plt.title('G')

plt.subplot(223)
plt.axis('off')
plt.imshow(img_b,cmap='gray')
plt.title('B')

img_rgb = Image.merge("RGB",[img_r,img_g,img_b])
plt.subplot(224)
plt.axis('off')
plt.imshow(img_rgb,cmap='gray')
plt.title('B')



plt.show()