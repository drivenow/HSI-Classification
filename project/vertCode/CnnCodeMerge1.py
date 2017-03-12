#from __future__ import print_function
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:14:25 2016
@author: Administrator
去掉relu激活函数，第三个位置识别， 35次迭代acc: 0.9018 - val_loss: 0.6912 - val_acc: 0.8025
"""
'''
用一个Graph模型训练验证码图片的四个位置；
图像处理:将验证码图片转化成灰度图像，图像背景是黑色。

Graph模型获取某一层的输出：
[model.inputs[i].input for i in model.input_order],model.outputs['conv'].get_output(train=False), on_unused_input='ignore')
'''


import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Model
from keras.layers import Input,Dense, Dropout, Activation, Flatten,merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
from keras.callbacks import EarlyStopping
from ImagePre import *

#%%
#outBasePath="D:/project/VertCode/mycode/output/model/"
#picBasePath="D:/project/VertCode/tuniu"
#labPath="D:/project/VertCode/1(1).txt"
outBasePath="G:/data/model/vertCode"
picBasePath="G:/data/vertcode/tuniu"
labPath="G:/data/vertcode/1(1).txt"
X_train,Y_train=getPicArrWithLab(picBasePath,labPath)

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
nb_epoch=1000

#%%         

"""
三步：权重共享结构+对每个位置分别训练一个Dense+四个output
merge_mode:sum，100次迭代准确率37%
merge_mode:concat，100次迭代准确率60~70%
"""
def getCNNGraphConcat(input_shape):
    input_layer = Input(shape=input_shape)
    conv1 = Convolution2D(32, kernel_size[0], kernel_size[1],
                                border_mode='valid',
                                dim_ordering="th")(input_layer)
    relu1 = Activation('relu')(conv1)
    conv2 = Convolution2D(32, kernel_size[0], kernel_size[1],
                                border_mode='valid')(relu1)
    relu2 = Activation('relu')(conv2)
    pool1 = MaxPooling2D(pool_size=pool_size)(relu2)
    drop1 =Dropout(0.25)(pool1)
    
    
    #第一个位置
    p1_conv1 = Convolution2D(32,kernel_size[0],kernel_size[1],activation="relu")(drop1)
    p1_flat1 = Flatten()(p1_conv1)
    p1_dense1 = Dense(128,activation="relu")(p1_flat1)
    p1_drop1 = Dropout(0.5)(p1_dense1)
    p1_output = Dense(nb_classes,activation="softmax")(p1_drop1)
    #第二个位置
    p2_conv1 = Convolution2D(32,kernel_size[0],kernel_size[1],activation="relu")(drop1)
    p2_flat1 = Flatten()(p2_conv1)
    p2_dense1 = Dense(128,activation="relu")(p2_flat1)
    p2_drop1 = Dropout(0.5)(p2_dense1)
    p2_output = Dense(nb_classes,activation="softmax")(p2_drop1)
    #第三个位置
    p3_conv1 = Convolution2D(32,kernel_size[0],kernel_size[1],activation="relu")(drop1)
    p3_flat1 = Flatten()(p3_conv1)
    p3_dense1 = Dense(128,activation="relu")(p3_flat1)
    p3_drop1 = Dropout(0.5)(p3_dense1)
    p3_output = Dense(nb_classes,activation="softmax")(p3_drop1)
    #第四个位置
    p4_conv1 = Convolution2D(32,kernel_size[0],kernel_size[1],activation="relu")(drop1)
    p4_flat1 = Flatten()(p4_conv1)
    p4_dense1 = Dense(128,activation="relu")(p4_flat1)
    p4_drop1 = Dropout(0.5)(p4_dense1)
    p4_output = Dense(nb_classes,activation="softmax")(p4_drop1)
    #输出
    all_output = merge(inputs=[p1_output,p2_output,p3_output,p4_output],mode="concat")
    
    model = Model(input=input_layer,output=all_output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


CNNGraphConcatModel = getCNNGraphConcat(input_shape)
from keras.utils.visualize_util import plot
plot(CNNGraphConcatModel,to_file="CNNGraphConcatModel.png")

#%% 
#early_stopping = EarlyStopping(monitor='val_loss', patience=2)
Y_train_concat=np.concatenate((Y_train[:,0,:],Y_train[:,1,:],Y_train[:,2,:],Y_train[:,3,:]),axis=1)
CNNGraphConcatModel.fit(X_train,Y_train_concat,
          batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, 
          validation_split=0.2,shuffle=True)
          
json_string=CNNGraphConcatModel.to_json()
open(outBasePath+'modelmerge'+'.json','w').write(json_string)
CNNGraphConcatModel.save_weights(outBasePath+"modelmerge"+"_weight.h5")
