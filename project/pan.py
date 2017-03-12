# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:48:08 2017
@author: Administrator
"""

#%% 读取数据
import numpy as np
basePath="D:/data/panSharping/"
#basePath = "G:/data/HSI/panSharping_li/"

def load2txt(path_prefix,file_num,keras_data):
    for i in range(file_num):
        filePath = path_prefix+str(i+1)+".txt"
        tmp_data=np.loadtxt(open(filePath),delimiter=",",skiprows=0)
        keras_data[:,:,i,0] = tmp_data
    return keras_data

keras_data = np.zeros((352836,7,7,1))
keras_label = np.zeros((352836,7,7,1))
keras_data = load2txt( basePath+"keras_data",7,keras_data)
keras_label = load2txt( basePath+"keras_label",7,keras_label)


#%%
from keras.models import Model
from keras.layers import Input,Convolution2D,merge
from keras.optimizers import sgd
import datetime

multi_Pan_input=Input(shape=(7,7,1))
conv1 = Convolution2D(nb_filter=64,nb_row=3,nb_col=3,activation="relu",border_mode="same")(multi_Pan_input)
conv2 = Convolution2D(nb_filter=64,nb_row=3,nb_col=3,activation="relu",border_mode="same")(conv1)
conv3 = Convolution2D(nb_filter=64,nb_row=3,nb_col=3,activation="relu",border_mode="same")(conv2)
conv4 = Convolution2D(nb_filter=64,nb_row=3,nb_col=3,activation="relu",border_mode="same")(conv3)
conv5 = Convolution2D(nb_filter=1,nb_row=3,nb_col=3,border_mode="same")(conv4)

merge_layer = merge([conv5,multi_Pan_input],mode='sum')

pan_model = Model(input=multi_Pan_input,output=merge_layer,name="pan_model")

SGD = sgd(lr=0.1,momentum=0.9,decay=0.0001)
pan_model.compile(loss="binary_crossentropy",optimizer=SGD,accuracy=['mse'])

pan_model.fit(keras_data,keras_label,batch_size=128,nb_epoch=40,validation_split=0.3)
datetime.datetime.now()
#%%
keras_data_predict = pan_model.predict(keras_data)
def save2txt(array4D):
    array_shape=array4D.shape
    for i in range(array_shape[2]):
        tmp_data=array4D[:,:,i,0]
        np.savetxt(basePath+"result/keras_data_predict"+str(i)+".txt", tmp_data, fmt="%f", delimiter=",")
save2txt(keras_data_predict)





