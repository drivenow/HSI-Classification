# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:16:11 2016

@author: Administrator
"""

import cv2
import numpy as np
from PIL import Image
import time
import math
import os


# 左侧边缘扫描
def getImgStBoard(imgVec):
    lstRowId=0
    bkFlag=False
    for y in range(0,80,1):
        for x in range(0,30,1):
            if imgVec[x][y]<230:
                bkFlag=True
                break
        if bkFlag == True:
            break
        lstRowId+=1
    return lstRowId
# 右侧边缘扫描
def getImgEdBoard(imgVec):
    lstRowId=79
    bkFlag=False
    for y in range(0,80,1):
        for x in range(0,30,1):
            if imgVec[x][79-y]<230:
                bkFlag=True
                break
        if bkFlag == True:
            break
        lstRowId-=1
    return lstRowId


def getSplitImg(imgVec,splitNum,imgOut):
    left=0
    right=len(imgVec[0])
    mid=int(np.ceil((right-left+0.0)/splitNum))
    for i in range(splitNum):
        imgOut.append(imgVec[:,max(left-3+mid*i,0):min(left+mid*i+mid+5,80)])
    return imgOut


def getAllSplitImg(imgVec):
    cnt_val=0
    left=getImgStBoard(imgVec)
    right=getImgEdBoard(imgVec)
    
    mid=(right-left)/4
    imgOut=[]
    #    for i in range(4):
    #        imgs.append(imgVec[:,max(left-2+mid*i,0):min(left+mid*i+mid+3,80)])
    
    #剔除中间的空白
    for y in range(int(left+3),int(right-2),1):
        flag=True
        cnt=1
        for x in range(0,30,1):
            if imgVec[x][y]<200:
                cnt=cnt+1
            if cnt>=cnt_val:    
                flag=False
                break   
        if flag == False:
            continue
        else: 
            #找到了空白的直线,找空白结束位置
            yend=y
            for yy in range(yend,right+1,1):
                cnt=0
                for xx in range(0,30,1):
                    if imgVec[xx][yy]<220:
                        cnt=cnt+1
                if cnt>=cnt_val:    
                    yend=yy
                    break
            #确定分割成的图像数    
            if 0.5*mid+left<y<=1.2*mid+left:
                splitNum=1
            elif 1.3*mid+left<y<=2.2*mid+left:
                splitNum=2
            elif 2.2*mid+left<y<=3.2*mid+left:
                splitNum=3
            else: 
                splitNum=4
            imgOut=getSplitImg(imgVec[:,left:y],splitNum,imgOut)
            if splitNum<4:
                imgOut=getSplitImg(imgVec[:,yend:right+1],4-splitNum,imgOut)
            break
    #如果没找到空白，直接分割
    if y==right-3:
        imgOut=getSplitImg(imgVec[:,left:right+1],4,imgOut)
    return imgOut

def depimg(img,depth):
    val=256/2**depth
    for i in range(len(img)):
        for j in range(len(img[0])):
            img[i][j]=img[i][j]/val
    return img
    
    
basePath="D:/project/VertCode/tuniu"
outBasePath="D:/project/VertCode/tuniuc"
outSplitPath="D:/project/VertCode/tuniuSplit"
files=os.listdir(basePath)

for fidx,f in enumerate(files):
    print fidx
    if fidx>10:break
    image = Image.open(basePath+"/"+f)
    fname=f.split(".")[0]
    image=image.convert("L")
    image=np.array(image)
  
    #将图像一切为4
#    imgVecs=[image]
    imgVecs=getSplitImg(image,4,imgVecs)
    
    #将图像反色
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j]>200:
                image[i][j]=255
            if image[i][j]<60:
                image[i][j]=0
            image[i][j]=255-image[i][j]    
#    cv2.imwrite('edges0.jpg',image)
 
    for idx,img in enumerate(imgVecs):      
        #去噪
        kernel = np.ones((2,2),np.float32)/4
        img = cv2.filter2D(img,-1,kernel)
        #cv2.GaussianBlur(img,(1,1),0)
        
        #二值化
#        img=depimg(img,2)#转化成位图       
#        ret1,img = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
#        ret2,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#        cv2.imwrite(outBasePath+"/"+fname+"_"+str(idx)+".jpg",img)
        
        #腐蚀
#        kernel = np.ones((2,2),np.uint8)
#        img1 = cv2.erode(img,kernel,iterations = 1)
#        img1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        ##img=cv2.medianBlur(img,3)
        cv2.imwrite(outSplitPath+'/'+str(fidx)+'.jpg',img)
        
#        for i in range(1):
#              #边缘检测
#            edges = cv2.Canny(image, 175, 320, apertureSize=3)
#        #            cv2.imshow('edge',edges)           
#            #去掉边缘
#            image = image-edges
##                img=cv2.bilateralFilter(img,3,1,1)
#            fidx=int(fname)*4+idx
#            cv2.imwrite(outSplitPath+'/'+str(fidx)+'.jpg',img)



    
    

