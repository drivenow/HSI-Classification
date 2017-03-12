# -*- coding:UTF-8 -*-
from skimage import morphology
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2


im = Image.open('D:/project/VertCode/tuniu/83.jpg')
image = im.convert('L')# 转化到亮度
image=np.array(image)

   
image=morphology.erosion(image,morphology.square(2))#腐蚀
image=morphology.dilation(image,morphology.square(2))#膨胀

kernel = np.ones((2,2),np.float32)/4
image = cv2.filter2D(image,-1,kernel)

plt.figure()
plt.imshow(image, cmap=plt.cm.gray)









