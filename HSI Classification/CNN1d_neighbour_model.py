# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 16:04:20 2017
@author: Administrator
"""
#logBasePath = "D:/data/mylog/KerasDL/"
#rootPath = r'D:/data/HSI'

rootPath = "G:/data/HSI"
logBasePath = "G:/data/mylog/KerasDL/CNN1d_neighbour_model"

test_size = 0.9
nb_epoch = 2000
nb_classes = 16
batch_size = 200

block_size = 7
input_shape = (200,block_size*block_size)

#%%
from HSIDatasetLoad import *
from keras.utils import np_utils
import numpy as np
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
from keras.layers import Input,merge,Dense,Dropout,Flatten,Convolution1D,MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
import tensorflow as tf

"""
2——D guass distribution
"""
def guass_dist(mu1, mu2, sig1, sig2, rho, window):
     sig1_2 = sig1**2
     sig2_2 = sig2**2
     filters = np.zeros(window**2)
     for i in range(window):
         for j in range(window):
             e = (i-mu1)**2/sig1_2+(j-mu2)**2/sig2_2-2*rho*(i-mu1)*(j-mu2)/sig1*sig2
             f = 1/(2*3.14*sig1*sig2*np.sqrt(1-rho**2))*np.exp(-1/(2*(1-rho**2))*e)
             filters[i*window+j] = f
     return filters
             
"""
手动1改gaussian_filter的窗口
"""
def gauss_filter(shape, name=None, dim_ordering='th'):
    assert len(shape)==4,"guass filter shape error, shape %d,%d,%d,%d" %(shape[0],shape[1],shape[2],shape[3])
#    assert shape[1]==1,"size one filter"
    filters = np.zeros(shape)
    for i in range(shape[-1]):
        filters[0,0,:,i] = guass_dist(1,1,1,1,0,7)
    return tf.Variable(filters, dtype=tf.float32, name=name)

        
"""
batch200,gauss_filter,conv3,block3,adadelta.
1036/1036 [==============================] - 4s - loss: 0.1366 - acc: 0.9923 - val_loss: 0.4995 - val_acc: 0.8825
batch200,gauss_filter,conv4,block3,adadelta.
1000,1036/1036 [==============================] - 6s - loss: 0.2877 - acc: 0.9305 - val_loss: 0.4829 - val_acc: 0.8765
batch200,gauss_filter,conv4,block3,adadelta.
Epoch 701/2000
1036/1036 [==============================] - 8s - loss: 0.4024 - acc: 0.8996 - val_loss: 0.4835 - val_acc: 0.8757
batch200,gauss_filter,conv5,block3,adadelta.
Epoch 666/2000
1036/1036 [==============================] - 8s - loss: 0.3767 - acc: 0.8938 - val_loss: 0.4977 - val_acc: 0.8593
"""


def get_CNN1d_model(input_shape, classify_output_num, my_optimizer):
    input_layer = Input(input_shape)
    conv1 = Convolution1D(1, 1, subsample_length=1,init=gauss_filter, 
                          border_mode = "valid")(input_layer)
#    conv2 = Convolution1D(12, 5, subsample_length=1,
#                          border_mode = "valid")(conv1)
#    conv3 = Convolution1D(24, 4, subsample_length=1,
#                      border_mode = "valid")(conv2)
#    conv4 = Convolution1D(36, 5, subsample_length=1,
#                      border_mode = "valid")(conv3)
#    conv5 = Convolution1D(48, 5, subsample_length=1,
#                      border_mode = "valid")(conv4)
    flat1 = Flatten()(conv1)
    
    
    x = Dense(512, activation='relu', W_regularizer=l2(0.1))(flat1)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu', W_regularizer=l2(0.1))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    output = Dense(classify_output_num, activation='softmax')(x)
    
    model = Model(input = input_layer,output=output)
    model.compile(optimizer = my_optimizer,metrics=['accuracy'],loss='categorical_crossentropy')
    return model

basic_model = get_CNN1d_model(input_shape, nb_classes, 'adadelta')

#%% callbacks
from keras.utils.visualize_util import plot
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,TensorBoard,CSVLogger,ModelCheckpoint

plot(basic_model,to_file=logBasePath+"/CNN1d_neighbour_model.png",show_shapes=True)


csvLogger = CSVLogger(filename = logBasePath+"/CNN1d_neighbour_model_csv.log")
reduce_lr = ReduceLROnPlateau(patience = 40, factor = 0.1, verbose = 1)
tensor_board = TensorBoard(log_dir = logBasePath, histogram_freq=0, write_graph=True, write_images=True)
ealystop = EarlyStopping(monitor='val_loss', patience = 300)
checkPoint = ModelCheckpoint(filepath=logBasePath+"/CNN1d_neighbour_model_check", monitor = "val_acc", mode = "max", save_best_only=True)


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


#%% 评估结果
#print("Best: %f using %s" % (grid_model.best_score_, grid_model.best_params_))
##Best: 0.892892 using {'batch_size': 20, 'my_optimizer': 'Adadelta', }
#all_result = grid_model.grid_scores_
#
#best_model = grid_model.best_estimator_.model
#metric_names = best_model.metrics_names
#metric_values = best_model.evaluate(X_test, Y_test)
#print('Test score: %f , Test accuracy: %f' % (metric_values[0], metric_values[1]))
##Test score: 0.232483 , Test accuracy: 0.932617




        