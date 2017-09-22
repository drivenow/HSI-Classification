# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 16:04:20 2017
@author: Administrator
"""
#logBasePath = "D:/data/mylog/KerasDL/"
#rootPath = r'D:/data/HSI'

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#指定了每个GPU进程中使用显存的上限，但它只能均匀作用于所有GPU，无法对不同GPU设置不同的上限
#以上的显存限制仅仅为了在跑小数据集时避免对显存的浪费而已
#但在实际运行中如果达到了这个阈值，程序有需要的话还是会突破这个阈值。
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

rootPath = "G:/data/HSI"
logBasePath = "G:/data/mylog/KerasDL/"

block_size = 7
test_size = 0.9
validate_size = 0.8
nb_epoch = 500
nb_classes = 16
batch_size = 32

#是否使用pca
use_pca = False
if use_pca ==True:
    n_components = 30  
else:
    n_components = 200
input_shape = (block_size,block_size,n_components,1)

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
#reshape(none,7,7,30,1)
X_data_nei = np.array([x.reshape(block_size,block_size,n_components,1) for x in X_data_nei])

Y_data = np_utils.categorical_probas_to_classes(Y_data)
X_train_nei,X_test_nei,Y_train,Y_test,idx_train,idx_test = HSI.datasetSplit(X_data_nei,Y_data,idx_data,16,test_size = test_size)


#%%
from keras.layers import MaxPooling3D,Input,Dense,Dropout,Flatten,Convolution3D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from mykeras.callbacks import MyProgbarLogger
from keras.callbacks import ReduceLROnPlateau
from keras.utils.visualize_util import plot
from keras.optimizers import adadelta
from keras.regularizers import l2
from random import randint

border_name = "same"
def CNN3d_model(input_shape, nb_classes):
    input_layer = Input(input_shape)
    
    conv1 = Convolution3D(16,3,3,1,activation="relu",
                          border_mode=border_name,W_regularizer=l2(l2_lr))(input_layer)
    bn1 = BatchNormalization(axis=-1)(conv1)
    
    conv2 = Convolution3D(32,3,3,1,W_regularizer=l2(l2_lr),
                          activation="relu",border_mode=border_name)(bn1)
    bn2 = BatchNormalization(axis=-1)(conv2)
#    pool2 = MaxPooling2D(pool_size=(2,2))(bn2)
    drop2 = Dropout(0.3)(bn2)
    
    conv3 = Convolution3D(32,3,3,1,W_regularizer=l2(l2_lr),
                          activation="relu",border_mode=border_name)(drop2)
    bn3 = BatchNormalization(axis=-1)(conv3)
    pool3 = MaxPooling3D(pool_size=(3,3,1))(bn3)
    drop3 = Dropout(0.3)(pool3)
     
    conv4 = Convolution3D(64,3,3,1,W_regularizer=l2(l2_lr),
                          activation="relu",border_mode=border_name)(drop3)
    bn4 = BatchNormalization(axis=-1)(conv4)
    pool4 = MaxPooling3D(pool_size=(2,2,1))(bn4)
    drop4 = Dropout(0.3)(pool4)
    
    conv5 = Convolution3D(8,1,1,1,
                          activation="relu",border_mode=border_name)(drop4)
    bn5 = BatchNormalization(axis=-1)(conv5)
    
    flat1 = Flatten()(conv5)
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


def generate_batch_data_random(x, y, batch_size):
    """逐步提取batch数据到显存，降低对显存的占用"""
    ylen = len(y)
    loopcount = ylen//batch_size#向下取整
    while (True):
        i = randint(0,loopcount)
        yield x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]
    
CNN3d_model = CNN3d_model(input_shape,nb_classes)


#%% fit model
plot(CNN3d_model,to_file=logBasePath+"CNN3d_pca_preserve_band_model_V2.png",show_shapes=True)
reduce_lr = ReduceLROnPlateau(patience=40)
myLogger = MyProgbarLogger(to_file=logBasePath+"CNN3d_pca_preserv_band_model_V2.log")
#CNN2d_model.fit(X_train_nei,Y_train,nb_epoch=nb_epoch,batch_size=batch_size,verbose=1, 
#          validation_data=[X_test_nei,Y_test],callbacks=[myLogger,reduce_lr])
train_gene = generate_batch_data_random(X_train_nei,Y_train,batch_size)
test_gene = generate_batch_data_random(X_test_nei,Y_test,batch_size)
CNN3d_model.fit_generator(train_gene,nb_epoch=nb_epoch,verbose=1, validation_data=test_gene,
                          nb_val_samples=(len(Y_test)//batch_size*batch_size),
                          samples_per_epoch=len(Y_train)//batch_size*batch_size, callbacks=[myLogger,reduce_lr])

    
    
    