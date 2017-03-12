# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 16:04:20 2017
@author: Administrator
"""
#logBasePath = "D:/data/mylog/KerasDL/"
#rootPath = r'D:/data/HSI'

rootPath = "G:/data/HSI"
logBasePath = "G:/data/mylog/KerasDL/"

block_size = 7
test_size = 0.9
validate_size = 0.8
nb_epoch = 1000
nb_classes = 16
batch_size = 32

#是否使用pca
use_pca = True
n_components = 30
if use_pca ==True:
    input_shape = (block_size,block_size,n_components)
else:
    input_shape = (block_size,block_size,200)

#%%
from HSIDatasetLoad import *
from keras.utils import np_utils

HSI = HSIData(rootPath)
X_data = HSI.X_data
Y_data = HSI.Y_data
data_source = HSI.data_source
idx_data = HSI.idx_data

l2_lr = 0.1

#是否使用PCA降维
if use_pca==True:
    data_source = HSI.PCA_data_Source(data_source,n_components=n_components)
    
X_data_nei = HSI.getNeighborData(data_source,idx_data,block_size)

Y_data = np_utils.categorical_probas_to_classes(Y_data)
X_train_nei,X_test_nei,Y_train,Y_test,idx_train,idx_test = HSI.datasetSplit(X_data_nei,Y_data,idx_data,16,test_size = test_size)


#%%
from keras.layers import MaxPooling2D,Input,Dense,Dropout,Flatten,Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from mykeras.callbacks import MyProgbarLogger
from keras.callbacks import ReduceLROnPlateau
from keras.utils.visualize_util import plot
from keras.optimizers import adadelta
from keras.regularizers import l2

border_name = "same"
def CNN2d_model(input_shape, nb_classes):
    input_layer = Input(input_shape)
    
    conv1 = Convolution2D(16,3,3,activation="relu",
                          border_mode=border_name,W_regularizer=l2(l2_lr))(input_layer)
    bn1 = BatchNormalization(axis=-1)(conv1)
    
    conv2 = Convolution2D(32,3,3,W_regularizer=l2(l2_lr),
                          activation="relu",border_mode=border_name)(bn1)
    bn2 = BatchNormalization(axis=-1)(conv2)
#    pool2 = MaxPooling2D(pool_size=(2,2))(bn2)
    drop2 = Dropout(0.3)(bn2)
    
    conv3 = Convolution2D(32,3,3,W_regularizer=l2(l2_lr),
                          activation="relu",border_mode=border_name)(drop2)
    bn3 = BatchNormalization(axis=-1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(bn3)
    drop3 = Dropout(0.3)(pool3)
     
    conv4 = Convolution2D(64,3,3,W_regularizer=l2(l2_lr),
                          activation="relu",border_mode=border_name)(drop3)
    bn4 = BatchNormalization(axis=-1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(bn4)
    drop4 = Dropout(0.3)(pool4)
    
#    conv5 = Convolution2D(8,kernel_dim1=1,kernel_dim2=1,
#                          activation="relu",border_mode=border_name)(drop4)
    
    flat1 = Flatten()(drop4)
    dense1 = Dense(256,activation="relu",W_regularizer=l2(l2_lr))(flat1)
    bn5 = BatchNormalization()(dense1)
    drop5 = Dropout(0.3)(bn5)
    
    dense2 = Dense(256,activation="relu",W_regularizer=l2(l2_lr))(drop5)
    bn6 = BatchNormalization()(dense2)
    drop6 = Dropout(0.3)(bn6)
    
    dense3 = Dense(nb_classes,activation="softmax")(drop6)
    
    model = Model(input = input_layer,output = dense3)
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    return model
    
CNN2d_model = CNN2d_model(input_shape,nb_classes)


#%% fit model
plot(CNN2d_model,to_file=logBasePath+"CNN2d_pca_model_V2_tmp.png",show_shapes=True)
reduce_lr = ReduceLROnPlateau(patience=40)
myLogger = MyProgbarLogger(to_file=logBasePath+"CNN2d_pca_model_V2_tmp.log")
CNN2d_model.fit(X_train_nei,Y_train,nb_epoch=nb_epoch,batch_size=batch_size,verbose=1, 
          validation_data=[X_test_nei,Y_test],callbacks=[myLogger,reduce_lr])

    
    
    