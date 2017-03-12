# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 16:04:20 2017
@author: Administrator
"""
logBasePath = "D:/data/mylog/KerasDL/"
rootPath = r'D:/data/HSI'

#rootPath = "G:/data/HSI"
#logBasePath = "G:/data/mylog/KerasDL/"

test_size = 0.7
validate_size = 0.8
nb_epoch = 2000
nb_classes = 16
batch_size = 32

block_size = 7
input_shape = (200,block_size*block_size)
#data_Augmentation
padding = 5
#mess_window = 5

#%%
from HSIDatasetLoad import *
from keras.utils import np_utils
#数据规范化
def data_standard(X_data):
    import numpy as np
    sample,block_size,block_size,band = X_data.shape
    new_X_data = np.zeros((sample,band,block_size*block_size))
    for i in range(sample):
        for row in range(block_size):
            for col in range(block_size):
                new_X_data[i,:,row*block_size+col] = X_data[i,row,col,:]
    return new_X_data

HSI = HSIData(rootPath)
X_data = HSI.X_data
Y_data = HSI.Y_data
data_source = HSI.data_source
idx_data = HSI.idx_data

X_data = HSI.getNeighborData(data_source=data_source,idx_data=idx_data,block_size=block_size)
X_data = data_standard(X_data)

Y_data = np_utils.categorical_probas_to_classes(Y_data)
X_train,X_test,Y_train,Y_test,idx_train,idx_test = HSI.datasetSplit(X_data,Y_data,idx_data,16,test_size = test_size)


#%%
"""
batch normalization: relu之前
"""
from keras.layers import Input,merge,Dense,Dropout,Flatten,Convolution1D,MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from mykeras.callbacks import MyProgbarLogger
from keras.utils.visualize_util import plot
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,TensorBoard,CSVLogger,ModelCheckpoint
from keras.initializations import glorot_normal

def get_CNN1d_model(input_shape, classify_output_num):
    input_layer = Input(input_shape)
    conv1 = Convolution1D(32,3,activation = "relu",
                          border_mode = "same")(input_layer)
    pool1 = MaxPooling1D()(conv1)
    bn1 = BatchNormalization(axis = -1)(pool1)
    
    conv2 = Convolution1D(64,5,activation = "relu",
                          border_mode = "same")(bn1)
    pool2 = MaxPooling1D()(conv2)
    bn2 = BatchNormalization(axis = -1)(pool2)
    
    conv3 = Convolution1D(64,5,activation = "relu",
                          border_mode = "same")(bn2)
    pool3 = MaxPooling1D()(conv3)
    drop3 = Dropout(0.3)(pool3)
    bn3 = BatchNormalization(axis = -1)(drop3)
    
    conv4 = Convolution1D(64,5,activation="relu",
                          border_mode = "same")(bn3)
    pool4 = MaxPooling1D()(conv4)
    drop4 = Dropout(0.3)(pool4)
    bn4 = BatchNormalization(axis = -1)(drop4)
    
    flat1 = Flatten()(bn4)
#    flat2 = Flatten()(input_layer)
#    
#    merge1 = merge([flat1,flat2], mode = "concat")
    
    x=Dense(512, activation='relu')(flat1)
    x=Dropout(0.3)(x)
    x=Dense(512, activation='relu')(x)
    x=Dropout(0.3)(x)
    output=Dense(classify_output_num,activation='softmax')(x)
    
    model = Model(input = input_layer,output=output)
    model.compile(optimizer='adadelta',metrics=['accuracy'],loss='categorical_crossentropy')
    return model

basic_model = get_CNN1d_model(input_shape,nb_classes)

#%% fit model
plot(basic_model,to_file=logBasePath+"CNN1d_modelV3_neighbour.png",show_shapes=True)
myLogger = MyProgbarLogger(to_file=logBasePath+"CNN1d_modelV3_neighbour.log")
csvLogger = CSVLogger(filename=logBasePath+"CNN1d_modelV3_neighbour.log")
reduce_lr = ReduceLROnPlateau(patience=50,factor = 0.3, verbose =1)
tensor_board = TensorBoard(log_dir = logBasePath+"board")
ealystop = EarlyStopping(monitor='val_loss',patience=200)
checkPoint = ModelCheckpoint(filepath=logBasePath+"basic_model2_check",save_best_only=True)

basic_model.fit(X_train,Y_train,nb_epoch=nb_epoch,batch_size=batch_size,verbose=1, 
          validation_data=[X_test,Y_test],callbacks=[
            myLogger, ealystop, tensor_board, csvLogger, reduce_lr,checkPoint])

#%% 中间层输出
input_layer = basic_model.input
layer_name = 'convolution1d_23'
#layer_name = 'averagepooling1d_16'
conv4_layer = basic_model.get_layer(name=layer_name).output
conv4_layer_model = Model(input_layer,conv4_layer)

conv_output = conv4_layer_model.predict(X_train)

#%%
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
sample,length,band = conv_output.shape

#%matplotlib qt5

for sam in range(2):
    sam = sam+200
    plt.figure()
    sub_row=8#子图的行数
    sub_col=8#子图的列数
    sub_i=0#子图的编号
    for ba in range(band):
        sub_i = sub_i+1
        plt.subplot(sub_row,sub_col,sub_i)
        plt.plot(conv_output[sam,:,ba])
    plt.show()
        