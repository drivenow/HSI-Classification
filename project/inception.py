# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 22:14:54 2017

@author: Shenjunling
"""

from keras.layers import MaxPooling2D,Input,Dense,Dropout,Flatten,Convolution2D,Reshape,merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from mykeras.callbacks import MyProgbarLogger
from keras.callbacks import ReduceLROnPlateau
from keras.utils.visualize_util import plot
from keras.optimizers import adadelta
from keras.regularizers import l2

def inception(input_layer):
    border_name = "same"
    conv1_0 = Convolution2D(32,1,1,
                            activation="relu",border_mode=border_name)(input_layer)
    conv2_1 = Convolution2D(48,1,1,
                            activation="relu",border_mode=border_name)(input_layer)
    conv2_2 = Convolution2D(64,3,3,
                            activation = "relu",border_mode=border_name)(conv2_1)
    
    conv3_1 = Convolution2D(16,1,1,
                            activation = "relu",border_mode=border_name)(input_layer)
    conv3_2 = Convolution2D(32,5,5,
                            activation = "relu",border_mode=border_name)(conv3_1)
    
    pool4_1 = MaxPooling2D((3,3),strides=1,border_mode=border_name)(input_layer)
    conv4_2 = Convolution2D(32,1,1,
                            activation = "relu",border_mode=border_name)(pool4_1)
    
    merge_out = merge(inputs=[conv1_0,conv2_2,conv3_2,conv4_2,],mode="concat")
    
    return merge_out