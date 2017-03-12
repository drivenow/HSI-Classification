# -*- coding: utf-8 -*-
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit

#%%dataset2
"""
"""
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

X_data,Y_data,data_source,idx_data=datasetLoad2()#未划分训练集测试集的数据(不包括背景点)
data_source_input=np.array([img.reshape(145,145,1) for img in data_source.transpose()])


#%% args
nb_epoch=500
#autoCoderCNN_input_shape=Input(shape=(145,145,1))
autoCoderCNN_input_shape=Input(shape=(148,148,1))
eachLayer1Node_3Layer_input_shape=Input(shape=(200,))
classify_output_num=16
early_stopping = EarlyStopping(monitor='val_loss', patience=100,verbose=1)
encoding_dim = 32


#%%
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
"""
autoencoder
encoder
decoder
‘valid’:image_shape - filter_shape + 1.即滤波器在图像内部滑动
 ‘full’ shape: image_shape + filter_shape - 1.允许滤波器超过图像边界
"""
def autoCoderCNN(input_shape):
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_shape)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)
    
    
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(16, 3, 3, activation='relu',border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(1, 3, 3, activation='relu', border_mode='same')(x)
    return decoded

autoCoderCNN_output=autoCoderCNN(autoCoderCNN_input_shape) 
autoencoder_model = Model(autoCoderCNN_input_shape, autoCoderCNN_output)
autoencoder_model.compile(optimizer='adadelta', loss='binary_crossentropy')

#将数据从(200,145,145,1)变为(200,148,148,1)
def Img4DExtend(data_source_input,dim1,dim2_weight,dim3_height,dim4,out_dim2_weight,out_dim3_height):
    data_source_input_tmp=np.zeros((200,148,148,1))
    for iidx,img in enumerate(data_source_input):
        img=img[:,:,0]
        img=np.concatenate((img,np.zeros((145,3)).reshape(145,3)),axis=1)
        img=np.concatenate((img,np.zeros((3,148)).reshape(3,148)),axis=0)
        data_source_input_tmp[iidx,:,:,0]=img
    data_source_input=data_source_input_tmp
    return data_source_input
    
data_source_input_extend = Img4DExtend(data_source_input,200,145,145,1,148,148)

autoencoder_model.fit(data_source_input_extend, data_source_input_extend,
                nb_epoch=200,
                batch_size=256,
                shuffle=True,
                validation_split=0.3)
"""
data_source_input_extend_decoded:经过编码解码后，得到的与原来数据集同大小的数据
"""
data_source_input_extend_decoded = autoencoder_model.predict(data_source_input_extend)

#%%
#将数据从(200,148,148,1)变为(200,145,145,1)
def Img4DCut(data_source_input):
    data_source_input_tmp=np.zeros((200,145,145,1))
    for iidx,img in enumerate(data_source_input):
        img=img[0:145,0:145,0]
        data_source_input_tmp[iidx,:,:,0]=img
    data_source_input=data_source_input_tmp
    return data_source_input
data_source_input_cut=Img4DCut(data_source_input_extend_decoded)
data_source_input_cut_X=np.array([img.reshape(21025) for img in data_source_input]).transpose()

                
                
#%% 多层模型
"""
三层，每层一个节点,迭代3000次，训练集0.92.验证集0.74
"""
def eachLayer1Node_3Layer(input_shape,classify_output_num):
    x=Dense(128, activation='relu')(input_shape)#30,0.33;50,0.85,100,0.36,500,0.96(收敛)
    x=Dense(64, activation='relu')(x)#50,0.03，100,0.70
#    x=Dropout(0.3)(x)
    x=Dense(32,activation='relu')(x)#50,0.71,；50,0.49；50,0.86；50.0.88；50,0.32,；100(未收敛)，500（收敛）
#    x=Dropout(0.3)(x)
    output=Dense(classify_output_num,activation='softmax')(x)
    return output


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

Y_data=np_utils.categorical_probas_to_classes(Y_data)
X_train,X_test,Y_train,Y_test,idx_train,idx_test=datasetSplit(data_source_input_cut_X,Y_data,idx_data,num_calss=16)


print ("start mlp classification...")
oneLayerOneNode_output=eachLayer1Node_3Layer(eachLayer1Node_3Layer_input_shape,classify_output_num)
eachLayer3Node_3Layer_model=Model(eachLayer1Node_3Layer_input_shape,output=oneLayerOneNode_output)
eachLayer3Node_3Layer_model.compile(optimizer='adadelta',metrics=['accuracy'],loss='categorical_crossentropy')
early_stopping = EarlyStopping(monitor='val_loss', patience=100,verbose=0)
eachLayer3Node_3Layer_model.fit(X_train,Y_train, nb_epoch=nb_epoch,validation_data=(X_test,Y_test),callbacks=[early_stopping])





