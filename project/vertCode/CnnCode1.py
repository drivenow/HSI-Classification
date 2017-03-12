#from __future__ import print_function
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:14:25 2016
@author: Administrator
"""
'''
用四个模型训练验证码图片的四个位置；
图像处理:将验证码图片转化成灰度图像，图像背景是黑色
'''
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.callbacks import EarlyStopping
from ImagePre import getPicArrWithLab


#%% 
picBasePath="D:/project/VertCode/tuniu"
labPath="D:/project/VertCode/1(1).txt"
X_train,Y_train=getPicArrWithLab(picBasePath,labPath,reverse=True,binary=False,skele=False)


#%%
# input image dimensions
img_rows, img_cols = 30, 80
# number of convolutional filters to use
nb_filters = 40
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
#
input_shape=(1,30,80)
#
nb_classes=36
#
batch_size=128
#
nb_epoch=100


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

def getCNN4Layer(input_shape,nb_classes):
    model = Sequential()
    
    model.add(Convolution2D(32, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape,dim_ordering="th"))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(64, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(32, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model

def getCNN3Layer(input_shape,nb_classes):
    model = Sequential()

    model.add(Convolution2D(32, 4, 4,
                            border_mode='valid',
                            input_shape=input_shape,dim_ordering="th"))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(36))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])
    return model
    
for ii in range(4):
    model=getCNN4Layer(input_shape,nb_classes)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(X_train[:,:,:,:], Y_train[:,ii,:], batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_split=0.2,shuffle=True,callbacks=[early_stopping])
#    score = model.evaluate(X_test, Y_test[:,ii,:], verbose=0)
#    print('Test score:', score[0])
#    print('Test accuracy:', score[1])
              
    outBasePath="D:/project/VertCode/mycode/output/model/"
    print ("saving model"+str(ii))
    json_string=model.to_json()
    open(outBasePath+'model'+str(ii)+'.json','w').write(json_string)
    model.save_weights(outBasePath+"model"+str(ii)+"_weight.h5")

