# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:26:53 2017

@author: Shenjunling
"""
#logBasePath = "D:/data/mylog/KerasDL/"
#rootPath = r'D:/data/HSI'

rootPath = "G:/data/HSI"
logBasePath = "G:/data/mylog/KerasDL/HSI_LSTM_model"

test_size = 0.9
nb_epoch = 2000
nb_classes = 16
batch_size = 200

block_size = 1
input_shape = (200,block_size*block_size)

#%%
from HSIDatasetLoad import *
from keras.utils import np_utils
import numpy as np
#数据规范化
def data_standard(X_data):
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
from keras.layers import Input,merge,Dense,Dropout,Flatten,Convolution1D,MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
import tensorflow as tf
from keras.layers import LSTM



def get_model(input_shape, classify_output_num, my_optimizer):
    input_layer = Input(input_shape)
    lstm = LSTM(100,activation="relu")(input_layer)
    
    
    x = Dense(256, activation='relu', W_regularizer=l2(0.1))(lstm)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu', W_regularizer=l2(0.1))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    output = Dense(classify_output_num, activation='softmax')(x)
    
    model = Model(input = input_layer,output=output)
    model.compile(optimizer = my_optimizer,metrics=['accuracy'],loss='categorical_crossentropy')
    return model

basic_model = get_model(input_shape, nb_classes, 'adadelta')

#%% callbacks
from keras.utils.visualize_util import plot
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,TensorBoard,CSVLogger,ModelCheckpoint

plot(basic_model,to_file=logBasePath+"/HSI_LSTM_model.png",show_shapes=True)


csvLogger = CSVLogger(filename = logBasePath+"/HSI_LSTM_model.log")
reduce_lr = ReduceLROnPlateau(patience = 40, factor = 0.1, verbose = 1)
tensor_board = TensorBoard(log_dir = logBasePath, histogram_freq=0, write_graph=True, write_images=True)
ealystop = EarlyStopping(monitor='val_loss', patience = 300)
checkPoint = ModelCheckpoint(filepath=logBasePath+"/model_check", monitor = "val_acc", mode = "max", save_best_only=True)


#%% fit model
basic_model.fit(X_train,Y_train,nb_epoch=nb_epoch,batch_size=30,verbose=1, 
          validation_data=[X_test,Y_test],callbacks=[ealystop, tensor_board, csvLogger, reduce_lr, checkPoint])

#%% fit model
#from keras.wrappers.scikit_learn import KerasClassifier
#keras_model = KerasClassifier(get_CNN1d_model, input_shape = input_shape, classify_output_num = nb_classes)#keras model
#
#from sklearn.grid_search import GridSearchCV
#
#param_grid = dict(
#    my_optimizer = ['Adadelta','RMSprop'],
#    batch_size = [10,20,30,50],
#    nb_epoch = [nb_epoch],
#    validation_data=[[X_test,Y_test]],
#    callbacks=[[ealystop, reduce_lr]],
#    verbose=[1]
#    )
#                                    
#grid_model = GridSearchCV(estimator = keras_model, param_grid = param_grid, n_jobs=1)#scoring=make_scorer(mean_squared_error)
#grid_model.fit(X_train, Y_train)




        