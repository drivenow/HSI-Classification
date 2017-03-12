# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:52:06 2017

@author: Administrator
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#%% 绘图
"""
import matplotlib.pyplot as plt
#保存图像
cv2.imwrite('split'+str(i)+'.jpg',img)

#显示结果
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax1.imshow(img, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('original', fontsize=20)

ax2.imshow(skeleton, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('skeleton', fontsize=20)

fig.tight_layout()
plt.show()
"""
#%% 读取图片
"""
image = Image.open("D:/project/VertCode/tuniu/718.jpg").convert("L")#图片对象
image=np.array(image)#图片数组

im=cv2.imread("D:/project/VertCode/tuniu/718.jpg",cv2.IMREAD_ANYDEPTH)
kernel = np.ones((2,2),np.float32)/4
im1=cv2.filter2D(im, 2, kernel)
"""

#%% 函数
"""
将图像减去边缘
removeNum:减去边缘的次数
"""
def getEdgeRemoved(image,removeNum):
    for i in range(removeNum):
        #边缘检测
        edges = cv2.Canny(image, 100, 100, apertureSize=3)
        cv2.imshow('edge',edges)              
         #去掉边缘
        for ii in range(len(edges)):
            for jj in range(len(edges[0])):
                if edges[ii][jj]!=0:
                    image[ii][jj]=0
        cv2.imshow('edge removed',image)
    return image


"""
标签文件1(1).txt：图片编号\t图片验证码
delimeter:编号和验证码的分隔符
return:标签字典[文件名：文件标签]
"""
def getLab(labPath,delimeter="\t"):
    labelFile=open(labPath,"r")#
    lines=labelFile.readlines() 
    labels={}
    for lidx,line in enumerate(lines):
#        print "getLab: "+line
        print ("getLab: "+line)
        assert len(line.split(delimeter))==2, "invalid filed length in line "+str(lidx)
        if line.strip()!="" and line is not None:
            strs=line.split("\t")
            labels[strs[0].strip()]=strs[1].strip()
    return labels
    
"""
实施骨架算法,要求输入是二值图像
"""
from skimage import morphology
def getSkele(imgArr,bigsize=1):
    skeleton =morphology.skeletonize(imgArr)#收缩
    skeleton=morphology.dilation(skeleton,morphology.square(bigsize))#膨胀
    #将true,false转化成0,1
    for i in range(len(skeleton)):
        for j in range(len(skeleton[0])):
            if skeleton[i][j]==True:
                imgArr[i][j]=1
            else:
                imgArr[i][j]=0
    return imgArr

"""
将图像反色，针对白色背景的图像，将其转换成黑色背景
threshold: 将灰度值低于该值的点，灰度值归为0
return: 反色后的图像
"""
def getColorReverse(imgArr,threshold=0):
    for i in range(len(imgArr)):
        for j in range(len(imgArr[0])):
            imgArr[i][j]=255-imgArr[i][j]
            if imgArr[i][j]<threshold:
                imgArr[i][j]=0
    return  imgArr

"""
将图像二值化,采用OTSU寻找阈值点
return:0,1值的图像
"""
def gray2binary(imgArr):
    ret2,imgArr = cv2.threshold(imgArr,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)#转化成二值图
#    imgArr只有0和255这两个值
    imgArr=imgArr/255
    return imgArr
    
    

"""
骨架算法skele要求图像必须是二值的
eg ：x_train,y_train=getPicArrWithLab("D:/project/VertCode/tuniu","D:/project/VertCode/1(1).txt")
"""
def getPicArrWithLab(picBasePath,labPath,labDeli="\t",reverse=True,binary=False,skele=False):
    labels=getLab(labPath)
    files=os.listdir(picBasePath)    
    x=[]
    y=[]
    for fidx,f in enumerate(files):
        if 1000<int(f.split(".")[0]):
#            print "readImg: "+f.split(".")[0]
            print ("readImg: "+f.split(".")[0])
            imgArr=cv2.imread(picBasePath+"/"+f,cv2.IMREAD_GRAYSCALE)
            if reverse==True:
                imgArr= getColorReverse(imgArr,70)
            if binary==True:
                imgArr=gray2binary(imgArr)
#                plt.imshow(imgArr)#显示图像      
            if skele==True:
                imgArr=getSkele(imgArr)
#                plt.imshow(imgArr)#显示图像
            
            x.append(np.array([imgArr]))#增加一维，波段维 
            
            ytmp=[]
            text=labels[f.split(".")[0]]#从图片名获取图片编号
            for i in range(4): 
                labs=np.zeros(36)#0-9,a-z,36个标记位置
                labs= [int(ii) for ii in labs]
                if 47<ord(text[i])<58:                
                    labs[ord(text[i])-48]=1
                    ytmp.append(labs)
                if 96<ord(text[i])<123:
                    labs[ord(text[i])-97+10]=1
                    ytmp.append(labs)               
            assert len(ytmp)==4, "invalid verify code length in line: "+str(fidx)
            y.append(ytmp)
    y=np.array(y)
    x=np.array(x).astype('float32')/255
    return x,y
    
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


def getSplitX(x,y,deltaPct):
    array=[]
    len=(y-x)/4
    py=int(len*deltaPct)
    array.append([x,x+len+py])
    array.append([x+len-py,x+len*2+py])
    array.append([x+len*2-py,x+len*3+py])
    array.append([x+len*3-py,y])
    return array

"""
#对imgVec均匀切分
getSplitImg()
imgVec,图像数组
splitNum,平均分割成的子图个数
imgOut,子图数组
"""
def getSplitImg(imgVec,splitNum,imgOut):
    left=0
    right=len(imgVec[0])
    mid=np.ceil((right-left)/splitNum)
    for i in range(splitNum):
        imgOut.append(imgVec[:,max(left-2+mid*i,0):min(left+mid*i+mid+5,80)])
    return imgOut

"""
找到空白的列，切分(由于墨迹断痕等原因，效果可能不好)
eg:#将图像一切为4  imgOut=getAllSplitImg(image)
"""
def getAllSplitImg(imgVec):
    cnt_val=1
    left=getImgStBoard(imgVec)
    right=getImgEdBoard(imgVec)
    
    mid=(right-left)/4
    imgOut=[]
    #    for i in range(4):
    #        imgs.append(imgVec[:,max(left-2+mid*i,0):min(left+mid*i+mid+3,80)])
    
    #剔除中间的空白
    for y in range(int(left+3),int(right-2),1):
        flag=True
        cnt=0
        for x in range(0,30,1):
            if imgVec[x][y]<220:
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
                    if imgVec[xx][yy]<190:
                        cnt=cnt+1
                if cnt>=cnt_val:    
                    yend=yy
                    break
            #确定分割成的图像数    
            if 0.5*mid+left<y<=1.3*mid+left:
                splitNum=1
            elif 1.3*mid+left<y<=2.2*mid+left:
                splitNum=2
            elif 2.2*mid+left<y<=3.2*mid+left:
                splitNum=3
            else: 
                splitNum=4
#            print y,splitNum
            print (y,splitNum)
            imgOut=getSplitImg(imgVec[:,left:y],splitNum,imgOut)
            if splitNum<4:
                imgOut=getSplitImg(imgVec[:,yend:right+1],4-splitNum,imgOut)
            break
    #如果没找到空白，直接分割
    if y==right-3:
        imgOut=getSplitImg(imgVec[:,left:right+1],4,imgOut)
    return imgOut




    

    

