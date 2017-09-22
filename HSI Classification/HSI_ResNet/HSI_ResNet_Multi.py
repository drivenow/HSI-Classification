# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 16:04:20 2017
@author: Administrator
"""
#logBasePath = "D:/data/mylog/KerasDL/"
#rootPath = r'D:/data/HSI'

rootPath = "G:/data/HSI"
logBasePath = "G:/data/mylog/KerasDL/CNN_multi_model"

block_size = 13
test_size = 0.9
#validate_size = 0.8
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

#原始谱信息
data_source1 = HSI.data_source
X_train = data_source1[idx_train]
X_train = X_train.reshape(len(X_train),200,1)
X_test = data_source1[idx_test]
X_test = X_test.reshape(len(X_test),200,1)
#每类去一个样本
"""
迁移学习，多路学习
"""


#%%
from keras.layers import MaxPooling2D,Input,Dense,Dropout,Flatten,Convolution2D,Activation,merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.utils.visualize_util import plot
from keras.optimizers import adadelta
from keras.regularizers import l2

def identity_block(x,nb_filter,kernel_size=3):
    k1,k2,k3 = nb_filter
    out = Convolution2D(k1,1,1)(x)
    out = BatchNormalization(axis= -1)(out)
    out = Activation('relu')(out)

    out = Convolution2D(k2,kernel_size,kernel_size,border_mode='same')(out)
    out = BatchNormalization(axis=-1)(out)
    out = Activation('relu')(out)

    out = Convolution2D(k3,1,1)(out)
    out = BatchNormalization(axis=-1)(out)


    out = merge([out,x],mode='sum')
    out = Activation('relu')(out)
    return out
    
def conv_block(x,nb_filter,kernel_size=3):
    k1,k2,k3 = nb_filter

    out = Convolution2D(k1,1,1)(x)
    out = BatchNormalization(axis= -1)(out)
    out = Activation('relu')(out)

    out = Convolution2D(k2,kernel_size,kernel_size,border_mode='same')(out)
    out = BatchNormalization(axis= -1)(out)
    out = Activation('relu')(out)

    out = Convolution2D(k3,1,1)(out)
    out = BatchNormalization(axis= -1)(out)

    x = Convolution2D(k3,1,1)(x)
    x = BatchNormalization(axis= -1)(x)

    out = merge([out,x],mode='sum')
    out = Activation('relu')(out)
    return out


from keras.layers import Convolution1D,MaxPooling1D
def get_CNN1d(input_layer):
    
    conv1 = Convolution1D(64,3,activation = "relu",
                          border_mode = "same")(input_layer)
    bn1 = BatchNormalization(axis = -1)(conv1)
    
    conv2 = Convolution1D(128,3,activation = "relu",
                          border_mode = "same")(bn1)
    bn2 = BatchNormalization(axis = -1)(conv2)
    drop2 = Dropout(0.3)(bn2)
    
    conv3 = Convolution1D(128,5,activation = "relu",
                          border_mode = "same")(drop2)
    bn3 = BatchNormalization(axis = -1)(conv3)
    
    
    conv4 = Convolution1D(128,5,activation="relu",
                          border_mode = "same")(bn3)
    bn4 = BatchNormalization(axis = -1)(conv4)
    drop4 = Dropout(0.3)(bn4)
    
    
    flat = Flatten()(drop4)

    return flat

    
def CNN2d_model(input_shape, nb_classes):
    input_tensor = Input(input_shape)
    input_layer = Input((200,1))
    res1 = conv_block(input_tensor,[64,64,256],3)
    
    
    flat1 = Flatten()(res1)
    flat2 = get_CNN1d(input_layer)
    flat = merge((flat1,flat2),mode="concat")
#    res2 = identity_block(res1,[64,64,256],3)
#    res3 = identity_block(res2,[128,128,256],3)
#    res4 = identity_block(res3,[128,128,256],3)
    
    
    dense1 = Dense(1024,activation="relu",W_regularizer=l2(l2_lr))(flat)
    bn5 = BatchNormalization()(dense1)
    drop5 = Dropout(0.3)(bn5)
    
    dense2 = Dense(1024,activation="relu",W_regularizer=l2(l2_lr))(drop5)
    bn6 = BatchNormalization()(dense2)
    drop6 = Dropout(0.3)(bn6)
    
    dense3 = Dense(nb_classes,activation="softmax")(drop6)
    
    model = Model(input = [input_tensor,input_layer],output = dense3)
    model.compile(loss='categorical_crossentropy',#categorical_crossentropy
                  optimizer="adadelta",
                  metrics=['accuracy'])
    return model
    
CNN2d_model = CNN2d_model(input_shape,nb_classes)


#%% fit model
"""
PCA,3*3领域，批规范化，l2范数
categorical_crossentropy,adadelta,pca,block9,test0.9,l2_lr = 0.1,batch_size = 32，res3
Epoch 1000/1000
1036/1036 [==============================] - 7s - loss: 0.0021 - acc: 1.0000 - val_loss: 0.3779 - val_acc: 0.9109
categorical_crossentropy,adadelta,pca,block11,test0.9,l2_lr = 0.1,batch_size = 32，res4
Epoch 1000/1000
1036/1036 [==============================] - 10s - loss: 0.0046 - acc: 0.9990 - val_loss: 0.2558 - val_acc: 0.9347
categorical_crossentropy,adadelta,pca,block7,test0.9,l2_lr = 0.1,batch_size = 32，res4
Epoch 400/1000
1036/1036 [==============================] - 6s - loss: 0.0067 - acc: 1.0000 - val_loss: 0.6330 - val_acc: 0.8607
categorical_crossentropy,adadelta,pca,block5,test0.9,l2_lr = 0.1,batch_size = 32，res4
Epoch 1000/1000
1036/1036 [==============================] - 5s - loss: 0.0055 - acc: 1.0000 - val_loss: 0.8763 - val_acc: 0.8356
categorical_crossentropy,adadelta,nopca,block5,test0.9,l2_lr = 0.1,batch_size = 32，res4
Epoch 276/1000
1036/1036 [==============================] - 11s - loss: 0.0051 - acc: 1.0000 - val_loss: 0.3141 - val_acc: 0.9149
categorical_crossentropy,adadelta,nopca,block13,test0.9,l2_lr = 0.1,batch_size = 32，res4
och 1000/1000
1036/1036 [==============================] - 11s - loss: 0.0026 - acc: 1.0000 - val_loss: 0.2811 - val_acc: 0.9285
"""
plot(CNN2d_model,to_file=logBasePath+"/CNN_multi_model.png",show_shapes=True)
reduce_lr = ReduceLROnPlateau(patience=40)
#myLogger = MyProgbarLogger(to_file=logBasePath+"/CNN2d_pca_model.log")
csvLog = CSVLogger(logBasePath+"/CNN_multi_model.log")
CNN2d_model.fit([X_train_nei,X_train],Y_train,nb_epoch=nb_epoch,batch_size=batch_size,verbose=1, 
          validation_data=[[X_test_nei,X_test],Y_test],callbacks=[csvLog,reduce_lr])

    
    
    