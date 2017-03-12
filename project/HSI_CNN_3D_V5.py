# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 16:04:20 2017
@author: Administrator
"""
#logBasePath = "D:/data/mylog/KerasDL/"
#rootPath = r'D:/data/HSI'

rootPath = "G:/data/HSI"
logBasePath = "G:/data/mylog/KerasDL/"

block_size = 9
test_size = 0.9
validate_size = 0.8
nb_epoch = 1000
nb_classes = 16
batch_size = 32

#是否使用pca
use_pca = False
if use_pca ==True:
    n_components = 30  
else:
    n_components = 200
input_shape = (block_size,block_size,n_components)

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
#reshape(none,7,7,30)
X_data_nei = np.array([x.reshape(block_size,block_size,n_components) for x in X_data_nei])

Y_data = np_utils.categorical_probas_to_classes(Y_data)
X_train_nei,X_test_nei,Y_train,Y_test,idx_train,idx_test = HSI.datasetSplit(X_data_nei,Y_data,idx_data,16,test_size = test_size)


#%%
from keras.layers import ZeroPadding2D,merge,MaxPooling2D,Input,Dense,Dropout,Flatten,Convolution2D,Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from mykeras.callbacks import MyProgbarLogger
from keras.callbacks import ReduceLROnPlateau
from keras.utils.visualize_util import plot
from keras.optimizers import adadelta
from keras.regularizers import l2
from mykeras.googlenet_custom_layers import LRN,PoolHelper

def inception_layer(input_layer):
    border_name = "same"
    conv1_0 = Convolution2D(32,1,1,
                            activation="relu",border_mode=border_name)(input_layer)
    conv2_1 = Convolution2D(48,3,3,
                            activation="relu",border_mode=border_name)(input_layer)
#    conv2_2 = Convolution2D(64,3,3,
#                            activation = "relu",border_mode=border_name)(conv2_1)
    
    conv3_1 = Convolution2D(16,5,5,
                            activation = "relu",border_mode=border_name)(input_layer)
#    conv3_2 = Convolution2D(32,5,5,
#                            activation = "relu",border_mode=border_name)(conv3_1)
    
    pool4_1 = MaxPooling2D((3,3),strides=(1,1),border_mode=border_name)(input_layer)
    conv4_2 = Convolution2D(32,1,1,
                            activation = "relu",border_mode=border_name)(pool4_1)
    
    merge_out = merge(inputs=[conv1_0,conv2_1,conv3_1,conv4_2,],mode="concat")
    
    return merge_out

border_name = "valid"
def CNN2d_model(input_shape, nb_classes):
    input_layer = Input(input_shape)
    
    conv1 = Convolution2D(32,3,3,W_regularizer=l2(l2_lr),
                          activation="relu",border_mode=border_name)(input_layer)
    bn1 = BatchNormalization(axis=-1)(conv1)
    
    inception1 = inception_layer(bn1)
    inception2 = inception_layer(inception1)
    
    
    flat1 = Flatten()(inception2)
    dense1 = Dense(1000,activation="relu",W_regularizer=l2(l2_lr))(flat1)
    bn5 = BatchNormalization()(dense1)
    drop5 = Dropout(0.3)(bn5)
    
    dense2 = Dense(1000,activation="relu",W_regularizer=l2(l2_lr))(drop5)
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
plot(CNN2d_model,to_file=logBasePath+"CNN2d_inception_model_V5.png",show_shapes=True)
reduce_lr = ReduceLROnPlateau(patience=40)
myLogger = MyProgbarLogger(to_file=logBasePath+"CNN2d_inception_model_V5.log")

CNN2d_model.fit(X_train_nei,Y_train,nb_epoch=nb_epoch,batch_size=batch_size,verbose=1, 
          validation_data=[X_test_nei,Y_test],callbacks=[myLogger,reduce_lr])

    
    
    