# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 14:18:52 2016
@author: Shenjunling
"""
#%% args
rootPath = "G:/data/HSI"
logBasePath = "G:/data/mylog/KerasDL/"

nb_epoch1 = 2000
nb_epoch2 = 800
test_size = 0.6
batch_size = 64
spectral_dim = 200
encoded_dim = 60
HSI_class= 16
block_size = 1

use_pca = False
n_components = 30
if use_pca ==True:
    input_shape = (block_size,block_size,n_components)
else:
    input_shape = (block_size,block_size, spectral_dim)
    
svm_switch = 0#0 classify for spectral info, 1 classify for encoded info

#%% (1.加载数据)
from HSIDatasetLoad import *
from keras.utils import np_utils
import numpy as np

HSI = HSIData(rootPath)
X_data = HSI.X_data
Y_data = HSI.Y_data
data_source = HSI.data_source
idx_data = HSI.idx_data

#是否使用PCA降维
if use_pca==True:
    data_source = HSI.PCA_data_Source(data_source,n_components=n_components)

X_data_nei = HSI.getNeighborData(data_source=data_source,idx_data=idx_data,block_size=block_size)
Y_data = np_utils.categorical_probas_to_classes(Y_data)
X_train_nei,X_test_nei,Y_train,Y_test,idx_train,idx_test = HSI.datasetSplit(X_data_nei,Y_data,idx_data,16,test_size = test_size)
X_train = data_source[idx_train]
X_test = data_source[idx_test]

#%% (2)自编码器
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

"""
categorical_crossentropy
‘valid’:image_shape - filter_shape + 1.即滤波器在图像内部滑动
‘full’ shape: image_shape + filter_shape - 1.允许滤波器超过图像边界
"""
def get_DenseAE_model(input_dim):
    input_layer = Input(shape=(input_dim,))
    x = Dense(100, activation='relu')(input_layer)
    x = Dense(100, activation='relu')(x)
    x = Dense(60, activation='relu')(x)
    encoded = Dense(60, activation='relu')(x)#第四层，包括一层输入层

    x = Dense(60, activation='relu')(encoded)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    decoded = Dense(input_dim, activation='relu')(x)
    
    model = Model(input=input_layer, output=decoded)
    
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics = ['mse'])  
    return model


X_train_nei_flat = np.reshape(X_train_nei, (-1,np.prod(input_shape)))
X_test_nei_flat = np.reshape(X_test_nei, (-1,np.prod(input_shape)))
spatial_model = get_DenseAE_model(np.prod(input_shape))
reduce_lr2 = ReduceLROnPlateau(monitor="val_loss",factor=0.1,patience=30)
early_stopping2 = EarlyStopping(monitor='val_loss', patience=50,verbose=1)

spatial_model.fit(X_train_nei_flat, X_train_nei_flat,
             nb_epoch = nb_epoch1,
             batch_size = batch_size,
             validation_split=0.3, callbacks = [early_stopping2,reduce_lr2])

#%% （3）encoded data
encoded_spatial_model = Model(input = spatial_model.input,
                           output = spatial_model.layers[4].output)
X_test_encoded = encoded_spatial_model.predict(X_test_nei_flat)
X_train_encoded = encoded_spatial_model.predict(X_train_nei_flat)

#%% （4）分类模型
from keras.layers import Dropout
from Evaluate import modelMetrics


#（1）dnn
def get_mlp_classify_model(input_dim,classify_output_num):
    input_layer = Input(shape=(input_dim,))
    x=Dense(256, activation='relu')(input_layer)
    x=Dropout(0.3)(x)
    x=Dense(256, activation='relu')(x)
    output=Dense(classify_output_num, activation='softmax')(x)
    
    model=Model(input_layer,output=output)
     
    model.compile(optimizer='adadelta',metrics=['accuracy'],loss='categorical_crossentropy')
    return model
    
mlp_classify_model = get_mlp_classify_model(encoded_dim, HSI_class)
reduce_lr3 = ReduceLROnPlateau(monitor="val_loss",patience=50)
early_stopping3 = EarlyStopping(monitor='val_loss', patience=80,verbose=1)

mlp_classify_model.fit(X_train_encoded,Y_train, nb_epoch=nb_epoch2 ,
                       validation_data=(X_test_encoded,Y_test),
                        callbacks=[early_stopping3, reduce_lr3])
mlp_classify_model_report, mlp_classify_model_acu = modelMetrics(
     mlp_classify_model,X_test_encoded,Y_test)


#   sklearn mlp
from sklearn.neural_network import MLPClassifier
params = [{'solver': 'adam', 'learning_rate_init': 0.0001}]
labels = ["adam"]
for label, param in zip(labels, params):
    print("training: %s" % label)
    mlp = MLPClassifier(hidden_layer_sizes=(256,256,16), verbose=1, batch_size = 40,
                        max_iter=1000, **param)#tol:Tolerance for the training loss.
    mlp.fit(X_train_encoded, np_utils.categorical_probas_to_classes(Y_train))
    print("Training set score: %f" % mlp.score(X_train_encoded, np_utils.categorical_probas_to_classes(Y_train)))
    print("Training set loss: %f" % mlp.loss_)
clf_report, clf_acu = modelMetrics(mlp,X_test_encoded,Y_test)



#   svm 
from sklearn.svm import SVC
svc = SVC(kernel="rbf", C=30000, verbose=True)
if svm_switch==0:
    svc.fit(X_train,np_utils.categorical_probas_to_classes(Y_train))
    svc_train_accu = svc.score(X_train, np_utils.categorical_probas_to_classes(Y_train))
    svc_report, svc_accu = modelMetrics(svc, X_test, Y_test)
elif svm_switch==1:
    svc.fit(X_train_encoded, np_utils.categorical_probas_to_classes(Y_train))
    svc_train_accu = svc.score(X_train_encoded,np_utils.categorical_probas_to_classes(Y_train))
    svc_report, svc_accu = modelMetrics(svc,X_test_encoded,Y_test)
print(svc_train_accu, svc_accu)





