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
input_shape = (200,1)
nb_epoch = 2000
nb_classes = 16
batch_size = 32

#data_Augmentation
padding = 10
mess_window = 10

#%%
from HSIDataLoad import *

X_data,Y_data,data_source,idx_data=datasetLoad2(rootPath)
Y_data=np_utils.categorical_probas_to_classes(Y_data)
X_train,X_test,Y_train,Y_test,idx_train,idx_test=datasetSplit(X_data,Y_data,idx_data,num_calss=16,test_size=test_size)
##数据增强
#X_train_add, Y_train_add = data_Augmentation(X_train, Y_train, padding,mess_window)
##数据集合并
#X_train = np.concatenate((X_train,X_train_add),axis = 0)
#Y_train = np.concatenate((Y_train,Y_train_add),axis = 0)

#数据规范化
X_train = np.array([x.reshape(200,1) for x in X_train])
X_test = np.array([x.reshape(200,1) for x in X_test])

#划分验证集
Y_test = np_utils.categorical_probas_to_classes(Y_test)
X_validate,X_test,Y_validate,Y_test,idx_validate,idx_test=datasetSplit(X_test,Y_test,idx_test,num_calss=16,test_size=validate_size)


#%%
"""
batch normalization: relu之后
"""
from keras.layers import Input,merge,Dense,Dropout,Flatten,Convolution1D,AveragePooling1D,up
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from mykeras.callbacks import MyProgbarLogger
from keras.utils.visualize_util import plot
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,TensorBoard,CSVLogger,ModelCheckpoint


def get_basic_model(input_shape, classify_output_num):
    input_layer = Input(input_shape)
    conv1 = Convolution1D(nb_filter=4,filter_length=3,activation = "relu",
                          border_mode = "same")(input_layer)
    pool1 = AveragePooling1D()(conv1)
    bn1 = BatchNormalization(axis = -1)(pool1)
    
    conv2 = Convolution1D(nb_filter=8,filter_length=5,activation = "relu",
                          border_mode = "same")(bn1)
    pool2 = AveragePooling1D()(conv2)
    bn2 = BatchNormalization(axis = -1)(pool2)
    
    conv3 = Convolution1D(nb_filter=16,filter_length=5,activation = "relu",
                          border_mode = "same")(bn2)
    pool3 = AveragePooling1D()(conv3)
    drop3 = Dropout(0.3)(pool3)
    bn3 = BatchNormalization(axis = -1)(drop3)
    
    conv4 = Convolution1D(nb_filter = 16, filter_length=5,activation="relu",
                          border_mode = "same")(bn3)
    bn4 = BatchNormalization(axis = -1)(conv4)
    
    
    y=Dense(512, activation='relu')(conv4)
    y=Dropout(0.3)(y)
    y=Dense(256, activation='relu')(y)
    y=Dropout(0.3)(y)
    y=Dense(512,activation='relu')(y)
    
    
    conv5 = Convolution1D(nb_filter=16,filter_length=5,activation = "relu",
                          border_mode = "same")(input_layer)
    pool5 = AveragePooling1D()(conv5)
    bn5 = BatchNormalization(axis = -1)(pool5)
    
    conv6 = Convolution1D(nb_filter=8,filter_length=5,activation = "relu",
                          border_mode = "same")(bn5)
    pool6 = AveragePooling1D()(conv6)
    bn6 = BatchNormalization(axis = -1)(pool6)
    
    conv7 = Convolution1D(nb_filter=16,filter_length=5,activation = "relu",
                          border_mode = "same")(bn6)
    pool7 = AveragePooling1D()(conv7)
    drop7 = Dropout(0.3)(pool7)
    bn7 = BatchNormalization(axis = -1)(drop7)
    
    conv8 = Convolution1D(nb_filter = 16, filter_length=5,activation="relu",
                          border_mode = "same")(bn7)
    bn4 = BatchNormalization(axis = -1)(conv8)
    
    
    flat1 = Flatten()(bn4)
    flat2 = Flatten()(input_layer)
    
    merge1 = merge([flat1,flat2], mode = "concat")
    
    x=Dense(512, activation='relu')(merge1)
    x=Dropout(0.3)(x)
    x=Dense(256, activation='relu')(x)
    x=Dropout(0.3)(x)
    x=Dense(128,activation='relu')(x)
    x=Dropout(0.3)(x)
    output=Dense(classify_output_num,activation='softmax')(x)
    
    model = Model(input = input_layer,output=output)
    model.compile(optimizer='adadelta',metrics=['accuracy'],loss='categorical_crossentropy')
    return model

basic_model = get_basic_model(input_shape,nb_classes)

#%% fit model
plot(basic_model,to_file=logBasePath+"basic_augmentation_model.png",show_shapes=True)
myLogger = MyProgbarLogger(to_file=logBasePath+"basic_model2.log")
csvLogger = CSVLogger(filename=logBasePath+"basic_model2_csv.log")
reduce_lr = ReduceLROnPlateau(patience=50,factor = 0.3, verbose =1)
tensor_board = TensorBoard(log_dir = logBasePath+"board")
ealystop = EarlyStopping(monitor='val_loss',patience=100)
checkPoint = ModelCheckpoint(filepath=logBasePath+"basic_model2_check",save_best_only=True)

basic_model.fit(X_train,Y_train,nb_epoch=nb_epoch,batch_size=batch_size,verbose=1, 
          validation_data=[X_validate,Y_validate],callbacks=[
            myLogger, ealystop, tensor_board, csvLogger, reduce_lr,checkPoint])

#%% 中间层输出
input_layer = basic_model.input
layer_name = 'convolution1d_44'
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
    sub_row=6#子图的行数
    sub_col=6#子图的列数
    sub_i=0#子图的编号
    for ba in range(band):
        sub_i = sub_i+1
        plt.subplot(sub_row,sub_col,sub_i)
        plt.plot(conv_output[sam,:,ba])
    plt.show()
        