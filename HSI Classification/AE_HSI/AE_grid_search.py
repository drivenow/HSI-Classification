# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 15:52:47 2017
@author: Administrator
"""


from keras.layers import Dense,Input
from keras.models import Model

#%% (1.加载数据)
from sklearn.cross_validation import StratifiedShuffleSplit
from keras.utils import np_utils
import numpy as np
#lab转换成one-hot编码
def datasetSplit(data,lab,num_calss):
    ssp = StratifiedShuffleSplit(lab,n_iter=1,test_size=0.90)
    for trainlab,testlab in ssp:
        print("train:\n%s\ntest:\n%s" % (trainlab,testlab))
    X_train=data[trainlab]
    X_test=data[testlab]
    Y_train=np_utils.to_categorical(lab[trainlab],num_calss)
    Y_test=np_utils.to_categorical(lab[testlab],num_calss)
    return X_train,X_test,Y_train,Y_test

#%%dataset2
def datasetLoad2():
    rootPath = r'D:/data/HSI'
    Xpath=rootPath+'/labeled_data.1.27.txt'
    Ypath=rootPath+'/data_label.1.27.txt'
    
    X_data = np.loadtxt(open(Xpath,"rb"),delimiter=",",skiprows=0,dtype=np.float)
    X_data=X_data.transpose()
    Y_data = np.loadtxt(open(Ypath,"rb"),delimiter=",",skiprows=0,dtype=np.int)
    Y_data=np_utils.to_categorical(Y_data-1,16)
    
    return X_data,Y_data
    
X_data,Y_data=datasetLoad2()
#Y_data = np_utils.categorical_probas_to_classes(Y_data)
#%% keras wrappers
from keras.models import Sequential

"""
KerasClassifier接受的模型函数，
(1)函数参数不能是keras模型参数中不包含的(存疑？)
(2)（metrics未放在参数列表，会报错illegal parameter）
（3）nb_epoch和batch_size可以放在grid_search中

"""
def autoCoderDense_7(optimizer,loss):    
    model=Sequential()
    model.add(Dense(128,input_dim=200, activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(16,activation="softmax"))
    model.compile(optimizer=optimizer, loss=loss,metrics=["accuracy"]) 
    return model

"""
keras对sklearn的包装类，可以用grid_search参数
参数传入的顺序：
（1）首先是 fit, predict, predict_proba, and score函数中的参数
（2）其次是KerasClassifier中定义的参数
（3）再次是默认参数
注意： grid_search的默认score是estimator的score
"""
from keras.wrappers.scikit_learn import KerasClassifier
AE_keras=KerasClassifier(build_fn=autoCoderDense_7, nb_epoch=2, verbose=1)

#%%  grid_search
"""
一般来说，在优化算法中包含epoch的数目是一个好主意，
因为每批（batch）学习量（学习速率）、每个epoch更新的数目（批尺寸）和 epoch的数量之间都具有相关性。
"""
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.scorer import mean_squared_error
from sklearn.metrics import make_scorer

#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
optimizer = ['Adadelta','RMSprop']
loss=['categorical_crossentropy']
batch_size=[10]

param_grid = dict(optimizer=optimizer,
                  loss=loss,
                  batch_size=batch_size)
                                    
grid_search = GridSearchCV(estimator=AE_keras, param_grid=param_grid, n_jobs=1)#scoring=make_scorer(mean_squared_error)

#%% 
"""
只能用于预测标签
The model is not configured to compute accuracy. 
You should pass `metrics=["accuracy"]` to the `model.compile()
"""
validator=grid_search.fit(X_data,Y_data)


#%% 评估结果
best_model = validator.best_estimator_.model
metric_names = best_model.metrics_names
metric_values = best_model.evaluate(X_data, Y_data)
print('\n')
for metric, value in zip(metric_names, metric_values):
    print(metric, ': ', value)
print("Best: %f using %s" % (validator.best_score_, validator.best_params_))
print(validator.grid_scores_)#打印每一组参数的结果


    
    
    
