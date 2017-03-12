from __future__ import print_function
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:14:25 2016

@author: Administrator
"""
'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''


import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import os
from PIL import Image
import cv2
import numpy as np
import random
from skimage import morphology,draw
import matplotlib.pyplot as plt


outBasePath="D:/project/VertCode/mycode/output/model/"


#文件改明
basePath="D:/project/VertCode/tuniu"
#basePath="G:/project/VertCode/genCode1"
files=os.listdir(basePath)

labelFile=open("D:/project/VertCode/1(1).txt","r")
lines=labelFile.readlines()

labels={}
for lidx,line in enumerate(lines):
    if line.strip()!="" and line is not None:
        strs=line.split("\t")
        labels[strs[0].strip()]=strs[1].strip()
        
        
#将途牛验证码读入
#basePath="D:/project/VertCode/genCode2"
basePath="D:/project/VertCode/tuniuSplit"
#basePath="G:/project/VertCode/genCode1"
files=os.listdir(basePath)

x=[]
y=[]
for idx,f in enumerate(files):
    if idx<4000:    
        img=cv2.imread(basePath+"/"+f,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(20,30))
#        cv2.imshow("dsa",img)
#        for i in range(len(img)):
#            for j in range(len(img[0])):
#                img[i][j]=255-img[i][j]
#                if img[i][j]<70:
#                    img[i][j]=0    
#                    
#        ret2,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)#转化成二值图
#        for i in range(len(img)):
#            for j in range(len(img[0])):
#                if img[i][j]==255:
#                    img[i][j]=1
#                if img[i][j]==0:
#                    img[i][j]=0
#        #实施骨架算法
#        skeleton =morphology.skeletonize(img)#输入是二值化图像
#        skeleton=morphology.dilation(skeleton,morphology.square(1))
#        for i in range(len(skeleton)):
#            for j in range(len(skeleton[0])):
#                if skeleton[i][j]==True:
#                    img[i][j]=255
#                else:
#                    img[i][j]=0
#                
#                
#        plt.imsave("D:/project/VertCode/checkcode/"+str(idx)+".png",img)
        x.append(np.array([img]))
        
        yy=[]
  
        labs=np.zeros(36)

        labs= [int(ii) for ii in labs]
        labs= [int(ii) for ii in labs]
        if 47<ord(labels[idx])<58:
            labs[ord(labels[idx])-48]=1
            yy.append(labs)
        if 96<ord(labels[idx])<123:
            labs[ord(labels[idx])-97+10]=1
            yy.append(labs)               
#        y.append(np.array(yy))
        y.append(yy)


y=np.array(y)
x=np.array(x).astype('float32') 

X_train=x
Y_train=y
X_train /= 255

batch_size = 64
nb_classes =36
nb_epoch = 200

# input image dimensions
img_rows, img_cols = 30, 20
# number of convolutional filters to use
nb_filters = 40
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
#
input_shape=(1,30,20)


ii=ii
model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape,dim_ordering="th"))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

#第一个位置的训练
#    model.fit(X_train, Y_train[:,ii,:], batch_size=batch_size, nb_epoch=nb_epoch,
#              verbose=1, validation_data=(X_test, Y_test[:,ii,:]))
model.fit(X_train[:,:,:,:], Y_train[:,ii,:], batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_split=0.2,shuffle=True)
#    score = model.evaluate(X_test, Y_test[:,ii,:], verbose=0)
#    print('Test score:', score[0])
#    print('Test accuracy:', score[1])

print ("saving model"+str(ii))
json_string=model.to_json()
open(outBasePath+'model.json','w').write(json_string)
model.save_weights(outBasePath+"model_weight.h5")
#    del model
    
    



# model.evaluate(test['data'],test['lab'], show_accuracy=True)
# model.predict_classes(test['data'])
