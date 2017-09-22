# -*- coding: utf-8 -*-
"""
Created on Tue Dec 06 14:06:47 2016

optimizer='adadelta',ada自适应的
metrics='accuracy',
loss='categorical_crossentropy'多类对数损失函数

@author: Administrator
"""

from keras.models import Sequential,Model
from keras.layers import Dense,Input,merge,Dropout
from keras.activations import softmax,relu
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit

#%% 数据读取
"""
X_data,Y_data,data_source,idx_data=datasetLoad2()#未划分训练集测试集的数据(不包括背景点)
Y_data=np_utils.categorical_probas_to_classes(Y_data)
X_train,X_test,Y_train,Y_test,idx_train,idx_test=datasetSplit(X_data,Y_data,idx_data,num_calss=16)
"""

def datasetSplit(data,lab,idx_of_data,num_calss):
    ssp = StratifiedShuffleSplit(lab,n_iter=1,test_size=0.90)
    for trainlab,testlab in ssp:
        print("train:\n%s\ntest:\n%s" % (trainlab,testlab))
    X_train=data[trainlab]
    X_test=data[testlab]
    Y_train=np_utils.to_categorical(lab[trainlab],num_calss)
    Y_test=np_utils.to_categorical(lab[testlab],num_calss)
    idx_train=idx_of_data[trainlab]
    idx_test=idx_of_data[testlab]
    return X_train,X_test,Y_train,Y_test,idx_train,idx_test


def datasetLoad2():
    rootPath = r'D:/data/HSI'
    Xpath=rootPath+'/labeled_data.1.27.txt'
    Ypath=rootPath+'/data_label.1.27.txt'
    imgPath=rootPath+'/data_source.1.27.txt'
    idxPath=rootPath+'/labeled_idx.1.27.txt'
    
    X_data = np.loadtxt(open(Xpath,"rb"),delimiter=",",skiprows=0,dtype=np.float)
    X_data=X_data.transpose()
    Y_data = np.loadtxt(open(Ypath,"rb"),delimiter=",",skiprows=0,dtype=np.int)
    Y_data=np_utils.to_categorical(Y_data-1,16)
    data_source=np.loadtxt(open(imgPath,"rb"),delimiter=",",skiprows=0,dtype=np.float)
    idx_data=np.loadtxt(open(idxPath,"rb"),delimiter=",",skiprows=0,dtype=np.int)
    idx_data=idx_data-1
    
    return X_data,Y_data,data_source,idx_data

#%% 模型评估
from keras.utils.np_utils import categorical_probas_to_classes
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#判断f1和总的准确率，输入的Y_test是概率形式的
def modelMetrics(model_fitted,X_test,Y_test):
    Y_predict_prob=model_fitted.predict(X_test)
    Y_predict_ctg=categorical_probas_to_classes(Y_predict_prob)
    report =classification_report(Y_predict_ctg,categorical_probas_to_classes(Y_test))#各个类的f1score
    accuracy = accuracy_score(Y_predict_ctg,categorical_probas_to_classes(Y_test))#总的准确度
    return report,accuracy
    
#%% args
X_data,Y_data,data_source,idx_data=datasetLoad2()#未划分训练集测试集的数据(不包括背景点)
Y_data=np_utils.categorical_probas_to_classes(Y_data)
X_train,X_test,Y_train,Y_test,idx_train,idx_test=datasetSplit(X_data,Y_data,idx_data,num_calss=16)

nb_epoch = 500
input_shape = (200,)#输入特征维度4
classify_output_num=16

"""
monitor='val_loss',需要监视的量
patience=2，监视两相比上一次迭代没有下降，经过几次epoch之后
在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练。
"""
early_stopping = EarlyStopping(monitor='val_loss', patience=50,verbose=1)

#%% 多层模型
"""
三层最典型多层感知机网络，每层单输入单输出,迭代3000次，训练集0.92.验证集0.74,新数据集训练集0.97，测试集0.74
"""
def basic_mlp_model(input_shape,classify_output_num):
    input_layer = Input(input_shape)
    x=Dense(128, activation='relu')(input_layer)
    x=Dropout(0.3)(x)
    x=Dense(64, activation='relu')(x)
    x=Dropout(0.3)(x)
    x=Dense(32,activation='relu')(x)
    x=Dropout(0.3)(x)
    output=Dense(classify_output_num,activation='softmax')(x)
    
    model = Model(input = input_layer,output=output)
    model.compile(optimizer='adadelta',metrics=['accuracy'],loss='categorical_crossentropy')
    return model

basic_mlp_model = basic_mlp_model(input_shape,classify_output_num)

#==============================================================================
#basic_mlp_model.fit(X_train,Y_train, nb_epoch=nb_epoch,validation_split=0.1,callbacks=[early_stopping])
basic_mlp_model.fit(X_train,Y_train, nb_epoch=nb_epoch,validation_data=(X_test,Y_test),callbacks=[early_stopping])
basic_mlp_model_report,basic_mlp_model_accuracy = modelMetrics(basic_mlp_model,X_test,Y_test)
#==============================================================================
#%% 一层多个节点
"""
一层三个节点，迭代971次，准确率在71~75%，验证集低约6个百分点,验证集低约5个百分点
"""
def eachLayer3Node_2Layer(input_shape,classify_output_num):
    dense1=Dense(64)
    x1=dense1(input_shape)
    x2=dense1(input_shape)
    x3=dense1(input_shape)
    assert dense1.get_output_at(0) == x1,"dense1层的第一个节点的输出不是x1"
    assert dense1.get_output_at(1) == x2,"dense1层的第二个节点的输出不是x2"
    assert dense1.get_output_at(2) == x3,"dense1层的第三个节点的输出不是x3"
    merged=merge([x1,x2,x3],mode='sum')
    output=Dense(16,activation='softmax')(merged)
    return output

"""
三层，前两层三个节点
"""
def eachLayer3Node_3Layer(input_shape,classify_output_num):
    dense1=Dense(64)
    x1=dense1(input_shape)
    x2=dense1(input_shape)
    x3=dense1(input_shape)
    assert dense1.get_output_at(0) == x1,"dense1层的第一个节点的输出不是x1"
    assert dense1.get_output_at(1) == x2,"dense1层的第二个节点的输出不是x2"
    assert dense1.get_output_at(2) == x3,"dense1层的第三个节点的输出不是x3"
    dense2=Dense(32)
    x11=dense2(x1)
    x21=dense2(x2)
    x31=dense2(x3)
    assert dense2.get_output_at(0) == x11,"dense2层的第一个节点的输出不是x1"
    assert dense2.get_output_at(1) == x21,"dense2层的第二个节点的输出不是x2"
    assert dense2.get_output_at(2) == x31,"dense2层的第三个节点的输出不是x3"
    merged=merge([x11,x21,x31],mode='sum')
    output=Dense(16,activation='softmax')(merged)
    return output

oneLayer3Node_output=eachLayer3Node_3Layer(input_shape,classify_output_num)



#%% 模型初始化
eachLayer3Node_3Layer_model=Model(input_shape,output=oneLayer3Node_output)

eachLayer3Node_3Layer_model.compile(optimizer='adadelta',metrics=['accuracy'],loss='categorical_crossentropy')



#==============================================================================
# eachLayer3Node_3Layer_model.fit(X_train,Y_train, nb_epoch=nb_epoch,validation_data=(X_test,Y_test),callbacks=[early_stopping])
# eachLayer3Node_3Layer_model_report,eachLayer3Node_3Layer_model_accuracy = modelMetrics(
#     eachLayer3Node_3Layer_model,X_test,Y_test)

#==============================================================================



