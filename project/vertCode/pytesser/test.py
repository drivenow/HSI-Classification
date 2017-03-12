# -*- coding: utf-8 -*-

import pytesseract
from PIL import Image
from numpy import *
import sys
import cv2
from matplotlib import pyplot as plt

fig1=plt.figure("fig1")
fig2=plt.figure("fig2")
fig3=plt.figure("fig3")






if __name__ == '__main__':
    
    basePath="D:/project/VertCode/tuniu"
#    im_p=sys.argv[1]
    im_p=basePath+"/2317.jpg"
    im=Image.open(im_p)
#    im = im.convert('L')
    im=array(im) 
    #剥离rgb
    im1=im[:,:,1]
    im2=im[:,:,2]
    im3=im[:,:,0]
    
#    ret2,im = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    #拉长图像，方便识别。
##    im = im.resize((200,80)) 
##    a = array(im) 
##    for i in xrange(len(a)): 
##        for j in xrange(len(a[i])): 
##          if a[i][j][0] == 255: 
##            a[i][j]=[0,0,0] 
##          else: 
##            a[i][j]=[255,255,255] 
#    im = Image.fromarray(im) 
#    im.show(False) 
#    
#    validateCode = pytesseract.image_to_string(im)
#    
#    print validateCode