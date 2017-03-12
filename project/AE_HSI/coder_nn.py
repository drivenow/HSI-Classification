# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 14:18:52 2016
@author: Shenjunling
"""

#%% (1.加载数据)
from HSIDataLoad import *

#dataset2
rootPath = r'G:/data/HSI'
X_data,Y_data,data_source,idx_data = datasetLoad2(rootPath)

#%% args
nb_epoch1 = 2000
nb_epoch2 = 2000
input_dim = 200
input_dim2 = 50
test_size = 0.9
batch_size1 = 64
classify_output_num = 16
encoding_dim = 256

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=50,verbose=1)


#%% (2)自编码器
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau

"""
‘valid’:image_shape - filter_shape + 1.即滤波器在图像内部滑动
 ‘full’ shape: image_shape + filter_shape - 1.允许滤波器超过图像边界
"""
def get_AEDense_model(input_dim, encoding_dim):
    # this is our input placeholder
    input_layer = Input(shape=(input_dim,))
    x = Dense(256, activation='relu')(input_layer)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    encoded = Dense(encoding_dim, activation='relu')(x)#第四层，包括一层输入层

    x = Dense(256, activation='relu')(encoded)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    decoded = Dense(input_dim, activation='relu')(x)
    
    model = Model(input=input_layer, output=decoded)
    
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics = ['mse'])
    
    return model

AEDense_model = get_AEDense_model(input_dim, encoding_dim)
reduce_lr = ReduceLROnPlateau(monitor="val_loss",patience=30)

AEDense_model.fit(X_data, X_data,
             nb_epoch = nb_epoch1,
             batch_size = batch_size1,
             validation_split=0.3,callbacks = [early_stopping,reduce_lr])

#%% (3)数据特征转化
encoded_output_model=Model(input = AEDense_model.input,
                           output = AEDense_model.layers[3].output)
X_data_encoded = encoded_output_model.predict(X_data)
X_data_reconstruct = AEDense_model.predict(X_data)
Y_data = np_utils.categorical_probas_to_classes(Y_data)
X_train,X_test,Y_train,Y_test,idx_train,idx_test = datasetSplit(X_data_encoded,Y_data,idx_data,num_calss=16,test_size=test_size)

#%% （4）分类模型
from keras.layers import Dropout
"""
三层，每层一个节点
"""
# 未挖掘出潜力的Dense_Model,这样分类效果不高
def get_mlp_classify_model(input_dim,classify_output_num):
    input_layer = Input(shape=(input_dim,))
    x=Dense(512, activation='relu')(input_layer)
    x=Dropout(0.3)(x)
    
    x=Dense(256, activation='relu')(x)
    x=Dropout(0.3)(x)
    
    x=Dense(128, activation='relu')(x)
    x=Dropout(0.3)(x)
    
    x=Dense(64, activation='relu')(x)
    x=Dropout(0.3)(x)
    output=Dense(classify_output_num, activation='softmax')(x)
    
    model=Model(input_layer,output=output)
    
    
    model.compile(optimizer='adadelta',metrics=['accuracy'],loss='categorical_crossentropy')
    return model
    
X_train,X_test,Y_train,Y_test,idx_train,idx_test = datasetSplit(X_data,Y_data,idx_data,num_calss=16,test_size=test_size)
mlp_classify_model = get_mlp_classify_model(input_dim, classify_output_num)
early_stopping = EarlyStopping(monitor='val_loss',patience=50,verbose=1)
mlp_classify_model.fit(X_train,Y_train, nb_epoch=nb_epoch2 ,
                       validation_data=(X_test,Y_test),callbacks=[early_stopping])

#
#Y_train = np_utils.categorical_probas_to_classes(Y_train)+1
#Y_test = np_utils.categorical_probas_to_classes(Y_test)+1
#
#from sklearn.neural_network import MLPClassifier
#from sklearn.metrics import classification_report
#from sklearn.metrics import accuracy_score
#clf=MLPClassifier(solver="lbfgs",alpha=1E-5,hidden_layer_sizes=(200,64,16),max_iter=5000)
#clf.fit(X_train,Y_train)
#
#Y_pred=clf.predict(X_test)
#report =classification_report(Y_pred,Y_test)
#acu=accuracy_score(Y_pred,Y_test)


#%% (5)模型评估
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

#==============================================================================
mlp_classify_model_report, mlp_classify_model_acu = modelMetrics(
     mlp_classify_model,X_test,Y_test)
#==============================================================================


#%% (6)中间层可视化
from keras.utils.visualize_util import plot
from keras.models import Model

#args
input_img=X_train[100,:]
model_to_visualize=autoencoder_model
"""
（1）将模型输出到文件
show_shapes：指定是否显示输出数据的形状，默认为False
show_layer_names:指定是否显示层名称,默认为True
"""
plot(model_to_visualize, to_file='D:/OneDrive/codes/python/nn/output/autocoder_HSI.png',show_shapes=True)

"""
(2)新建一个Model,指定input和output.(利用model0.get_layer函数,获得指定层）
"""
layer_name = 'convolution2d_2'
input_layer2=model_to_visualize.input
output_layer2=model_to_visualize.get_layer(layer_name).output
intermediate_layer_model = Model(input=input_layer2,output=output_layer2)
intermediate_output2 = intermediate_layer_model.predict(input_img)



