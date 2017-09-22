# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 16:04:20 2017
@author: Administrator
"""
logBasePath = "D:/data/mylog/KerasDL/"
rootPath = r'D:/data/HSI'
import time
start = time.clock()

#rootPath = "G:/data/HSI"
#logBasePath = "G:/data/mylog/KerasDL/HSI_resNet"

block_size = 13
test_size = 0.9
#validate_size = 0.8
nb_epoch = 600
nb_classes = 16
batch_size = 25
    
#def parse_args():
#    import argparse
#    parser = argparse.ArgumentParser()
#    parser.add_argument("-block_size",help="HSI block window")
#    parser.add_argument("-batch_size",help="trainnig batch")
#    parser.add_argument("-test_size")
#    parser.set_defaults(test_size=0.9)
#    args = parser.parse_args()
#    return args
#args = parse_args()
#print(args)
#block_size = int(args.block_size)
#batch_size = int(args.batch_size)
#test_size = float(args.test_size)

#是否使用pca
use_pca = True
n_components = 30
if use_pca ==True:
    input_shape = (block_size,block_size,n_components)
else:
    input_shape = (block_size,block_size,200)


#%%
from HSIDatasetLoad import *
from keras.utils import np_utils

HSI = HSIData(rootPath,"Pavia")
X_data = HSI.X_data
Y_data = HSI.Y_data
data_source = HSI.data_source
idx_data = HSI.idx_data

l2_lr = 0.1
#是否使用PCA降维
if use_pca==True:
    data_source = HSI.PCA_data_Source(data_source,n_components=n_components)
    
X_data_nei = HSI.getNeighborData(data_source,idx_data,block_size)

Y_data = np_utils.categorical_probas_to_classes(Y_data)
X_train_nei,X_test_nei,Y_train,Y_test,idx_train,idx_test = HSI.datasetSplit(X_data_nei,Y_data,idx_data,16,test_size = test_size)


#%%
from keras.layers import MaxPooling2D,Input,Dense,Dropout,Flatten,Convolution2D,Activation,merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.utils.visualize_util import plot
from keras.optimizers import adadelta
from keras.regularizers import l2
from keras.initializations import glorot_normal

def identity_block(x,nb_filter,kernel_size=3):
    k1,k2,k3 = nb_filter
    out = Convolution2D(k1,1,1)(x)
    out = BatchNormalization(axis= -1)(out)
    out = Activation('relu')(out)

    out = Convolution2D(k2,kernel_size,kernel_size,border_mode='same')(out)
    out = BatchNormalization(axis=-1)(out)
    out = Activation('relu')(out)

    out = Convolution2D(k3,1,1)(out)
    out = BatchNormalization(axis=-1)(out)


    out = merge([out,x],mode='sum')
    out = Activation('relu')(out)
    return out
    
def conv_block(x,nb_filter,kernel_size=3):
    k1,k2,k3 = nb_filter

    out = Convolution2D(k1,1,1)(x)
    out = BatchNormalization(axis= -1)(out)
    out = Activation('relu')(out)

    out = Convolution2D(k2,kernel_size,kernel_size,border_mode='same')(x)
    out = BatchNormalization(axis= -1)(out)
    out = Activation('relu')(out)

    out = Convolution2D(k3,1,1)(out)
    out = BatchNormalization(axis= -1)(out)
    out = MaxPooling2D()(out)

    x = Convolution2D(k3,1,1)(x)
    x = BatchNormalization(axis= -1)(x)
    x = MaxPooling2D()(x)

    out = merge([out,x],mode='sum')
    out = Activation('relu')(out)
    return out

border_name = "same"
def get_CNN2d_model(input_shape, classify_output_num):
    input_tensor = Input(input_shape)
    
    res1 = conv_block(input_tensor,[64,64,256],3)
#    res2 = identity_block(res1,[64,64,256],3)
#    res3 = identity_block(res2,[128,128,256],3)
#    res4 = identity_block(res3,[128,128,256],3)
    
    flat1 = Flatten()(res1)
    dense1 = Dense(1024,activation="relu",W_regularizer=l2(l2_lr))(flat1)
    bn5 = BatchNormalization()(dense1)
    drop5 = Dropout(0.3)(bn5)
    
    dense2 = Dense(1024,activation="relu",W_regularizer=l2(l2_lr))(drop5)
    bn6 = BatchNormalization()(dense2)
    drop6 = Dropout(0.3)(bn6)
    
    dense3 = Dense(nb_classes,activation="softmax")(drop6)
    
    model = Model(input = input_tensor,output = dense3)
    model.compile(loss='categorical_crossentropy',#categorical_crossentropy
                  optimizer="RMSprop",
                  metrics=['accuracy'])
#    model.compile(loss='categorical_crossentropy',#categorical_crossentropy
#                  optimizer="adadelta",
#                  metrics=['accuracy'])
    return model
    



#%% fit model
"""
RMSprop：

9880,500次迭代9870，后面缓慢上升，有触到9889
categorical_crossentropy,残差特征图128而不是256,RMSprop,pca15,block11,test0.9,l2_lr = 0.1,batch_size = 25，res2
9864,
categorical_crossentropy,RMSprop,pca15,block21,test0.9,l2_lr = 0.1,batch_size = 25，res1
9838
categorical_crossentropy,RMSprop,pca15,block11,test0.9,l2_lr = 0.1,batch_size = 25，res1
9833
categorical_crossentropy,RMSprop,pca15,block13,test0.9,l2_lr = 0.1,batch_size = 25，res1(残差模块pooling)1763.8
9885,9815,9841,9877,9893
categorical_crossentropy,RMSprop,pca15,block13,test0.9,l2_lr = 0.1,batch_size = 25，res1(残差模块pooling,残差层后跟卷积层)2527s
9862
categorical_crossentropy,RMSprop,pca15,block13,test0.9,l2_lr = 0.1,batch_size = 25，res1(残差模块pooling,64,64,64)2527s
9812
categorical_crossentropy,RMSprop,pca15,block13,test0.9,l2_lr = 0.1,batch_size = 25，res1(残差模块pooling,64,64,128)2527s
9796
categorical_crossentropy,RMSprop,pca15,block11,test0.9,l2_lr = 0.1,batch_size = 25，res1(残差模块pooling)1815s
9810
categorical_crossentropy,RMSprop,pca15,block11,test0.9,l2_lr = 0.1,batch_size = 25，res1(残差模块pooling,残差层后跟卷积层)2527s
9889
categorical_crossentropy,RMSprop,pca15,block7,test0.9,l2_lr = 0.1,batch_size = 25，res1(残差模块pooling,)1639s
9759
categorical_crossentropy,RMSprop,pca15,block9,test0.9,l2_lr = 0.1,batch_size = 25，res1(残差模块pooling,)1639s
9832
"""
#plot(CNN2d_model,to_file=logBasePath+"/CNN2d_pca_model.png",show_shapes=True)

#myLogger = MyProgbarLogger(to_file=logBasePath+"/CNN2d_pca_model.log")
reduce_lr = ReduceLROnPlateau(patience = 50, verbose =1)
ealystop = EarlyStopping(monitor='val_loss',patience =100)
CNN2d_model = get_CNN2d_model(input_shape,nb_classes)
csvLog = CSVLogger(logBasePath+"/"+str(batch_size)+".log")
#CNN2d_model.fit(X_train_nei,Y_train,nb_epoch=nb_epoch,batch_size=batch_size,verbose=1, 
#          validation_data=[X_test_nei,Y_test],callbacks=[csvLog,reduce_lr,ealystop])
CNN2d_model.fit(X_train_nei,Y_train,nb_epoch=nb_epoch,batch_size=batch_size,verbose=1, 
          validation_split=0.1,callbacks=[csvLog,reduce_lr,ealystop])

end = time.clock()
print ("read: %f s" % (end - start))

#%% fit model
#from keras.wrappers.scikit_learn import KerasClassifier
#keras_model = KerasClassifier(get_CNN2d_model, input_shape = input_shape, classify_output_num = nb_classes)#keras model
#
#from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import StratifiedShuffleSplit
#
#param_grid = dict(
#    batch_size = [20,25,30,35,40,45],
#    nb_epoch = [nb_epoch],
##    validation_split=[0.001],
#    validation_data=[[X_test_nei,Y_test]],
#    callbacks=[[ealystop, reduce_lr]],
#    )
#
#ssp = StratifiedShuffleSplit(Y_data,n_iter=1,test_size=test_size)                                    
#grid_model = GridSearchCV(estimator = keras_model, param_grid = param_grid, n_jobs=1,cv=ssp,refit=False)#scoring=make_scorer(mean_squared_error)
#Y_train = np_utils.to_categorical(Y_train, nb_classes)
#grid_model.fit(X_data_nei, Y_data)


#%% 写入详细分类结果
from util import cateAccuracy
def sample_count(Y):
    sample_count_train = {}
    if len(Y.shape)!=1:
        #转换onehot编码
        Y=np_utils.categorical_probas_to_classes(Y)
    for i in set(Y):
        sample_count_train[i] = list(Y).count(i)
    return sample_count_train
    
total_accu,accu = cateAccuracy(CNN2d_model,X_test_nei,Y_test)
train_count = sample_count(Y_train)
test_count = sample_count(Y_test)


f = open("pavia_result13_30_rmsprop_99.txt","a")
f.writelines(str(total_accu)+"\n")
for i in range(len(accu)):
    line = str(accu[i])+","+str(train_count[i])+","+str(test_count[i])+"\n"
    f.writelines(line)
    
f.write("=========================================================\n")
f.close()

#pred = CNN2d_model.predict(X_test_nei,batch_size=1)
#from sklearn.metrics import classification_report,accuracy_score
#accu = accuracy_score(np_utils.categorical_probas_to_classes(pred),np_utils.categorical_probas_to_classes(Y_test))
#from visualize import plot_result
#plot_result(Y_data,idx_data)#graoud_truth
#plot_train(Y_data,idx_data,idx_train)#train
#plot_result(np_utils.categorical_probas_to_classes(pred),idx_test)#test_result
#
#np.savetxt("Y_data.txt",Y_data)
#np.savetxt("idx_data.txt",idx_data)
#np.savetxt("idx_test.txt",idx_test)
#np.savetxt("idx_train.txt",idx_train)
#np.savetxt("pred.txt",pred)
    
    
    