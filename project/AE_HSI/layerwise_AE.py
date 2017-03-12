# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:59:40 2017
逐层初始化
@author: Shenjunling
"""
#%% (1.加载数据)
from HSIDataLoad import *
import numpy as np

#dataset2
rootPath = r'G:/data/HSI'
X_data,Y_data,data_source,idx_data = datasetLoad2(rootPath)
idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
X_data = X_data[idx]
Y_data = Y_data[idx]


#%% args
nb_epoch1 = 2000
nb_epoch2 = 2000
input_dim = 200
input_dim2 = 50
test_size = 0.5
batch_size1 = 256
classify_output_num = 16
encoding_dim = 64

from keras.callbacks import EarlyStopping



#%% (2)自编码器
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau
"""
‘valid’:image_shape - filter_shape + 1.即滤波器在图像内部滑动
 ‘full’ shape: image_shape + filter_shape - 1.允许滤波器超过图像边界
"""
def layerwise_model(input_layer, output_layer, X_data):
    
    model = Model(input = input_layer, output = output_layer)
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics = ['mse'])
    
    reduce_lr = ReduceLROnPlateau(monitor="val_loss",patience=30)
    early_stopping = EarlyStopping(monitor='val_loss', patience=100,verbose=1)
    model.fit(X_data, X_data,
             nb_epoch = nb_epoch1,
             batch_size = batch_size1,
             validation_split=0.3,callbacks = [early_stopping, reduce_lr])
    
    return model
    
input_layer = Input(shape=(200,))
encoded_layer = Dense(128,activation="relu")(input_layer)
output_layer = Dense(200,activation="relu")(encoded_layer)
input_data = X_data
model = layerwise_model(input_layer, output_layer,input_data)

