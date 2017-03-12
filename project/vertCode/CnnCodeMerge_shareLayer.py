# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:20:44 2017
@author: Administrator
getCNNGraphShare训练20 epoch：差强人意
val_dense_22_loss_4: 2.4789 - val_dense_22_acc_1: 0.4900 - val_dense_22_acc_2: 0.2225 - val_dense_22_acc_3: 0.2200 - val_dense_22_acc_4: 0.1650
"""

import numpy as np
np.random.seed(1337)  # for reproducibility
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
nb_epoch = 20

#%%
from keras.models import Model
from keras.layers import Input,Dense, Dropout, Activation, Flatten,merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils.visualize_util import plot
"""
shared_node, 四个输入，四个输出。输入输出一一对应，经过同一个模型训练
"""
def getCNNGraphShare(input_shape):
    p1_input_layer = Input(input_shape)
    p2_input_layer = Input(input_shape)
    p3_input_layer = Input(input_shape)
    p4_input_layer = Input(input_shape)
    
    shared_conv1 = Convolution2D(32, kernel_size[0], kernel_size[1],
                            border_mode='valid',activation="relu",dim_ordering="th")
    p1_shared_conv1 = shared_conv1(p1_input_layer)
    p2_shared_conv1 = shared_conv1(p2_input_layer)
    p3_shared_conv1 = shared_conv1(p3_input_layer)
    p4_shared_conv1 = shared_conv1(p4_input_layer)
    
    shared_conv2 = Convolution2D(32, kernel_size[0], kernel_size[1],
                                border_mode='valid',activation="relu",dim_ordering="th")
    p1_shared_conv2 = shared_conv2(p1_shared_conv1)
    p2_shared_conv2 = shared_conv2(p2_shared_conv1)
    p3_shared_conv2 = shared_conv2(p3_shared_conv1)
    p4_shared_conv2 = shared_conv2(p4_shared_conv1)
   
    shared_pool1 = MaxPooling2D(pool_size=pool_size)
    p1_shared_pool1 = shared_pool1(p1_shared_conv2)
    p2_shared_pool1 = shared_pool1(p2_shared_conv2)
    p3_shared_pool1 = shared_pool1(p3_shared_conv2)
    p4_shared_pool1 = shared_pool1(p4_shared_conv2)
    
    shared_drop1 = Dropout(0.25)
    p1_shared_drop1 = shared_drop1(p1_shared_pool1)
    p2_shared_drop1 = shared_drop1(p2_shared_pool1)
    p3_shared_drop1 = shared_drop1(p3_shared_pool1)
    p4_shared_drop1 = shared_drop1(p4_shared_pool1)
    
    shared_flat1 = Flatten()
    p1_shared_flat1 = shared_flat1(p1_shared_drop1)
    p2_shared_flat1 = shared_flat1(p2_shared_drop1)
    p3_shared_flat1 = shared_flat1(p3_shared_drop1)
    p4_shared_flat1 = shared_flat1(p4_shared_drop1)
    
    dense1 = Dense(128,activation="relu")
    p1_dense1 = dense1(p1_shared_flat1)
    p2_dense1 = dense1(p2_shared_flat1)
    p3_dense1 = dense1(p3_shared_flat1)
    p4_dense1 = dense1(p4_shared_flat1)
    
    
    dense2 = Dense(nb_classes,activation="softmax")
    p1_dense2 = dense2(p1_dense1)
    p2_dense2 = dense2(p2_dense1)
    p3_dense2 = dense2(p3_dense1)
    p4_dense2 = dense2(p4_dense1)
    

    model = Model(input=[p1_input_layer,p2_input_layer,p3_input_layer,p4_input_layer],
                  output = [p1_dense2,p2_dense2,p3_dense2,p4_dense2])
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model

CNNGraphShareModel=getCNNGraphShare(input_shape)
plot(CNNGraphShareModel,to_file="CNNGraphShareModel.png")

#%% fit 
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
CNNGraphShareModel.fit([X_train,X_train,X_train,X_train],
         [Y_train[:,0,:],Y_train[:,1,:],Y_train[:,2,:],Y_train[:,3,:]],
          batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, 
          validation_split=0.2,shuffle=True)