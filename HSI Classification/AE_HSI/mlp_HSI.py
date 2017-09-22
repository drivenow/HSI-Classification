# -*- coding: utf-8 -*-
"""
Created on Tue Dec 06 14:06:47 2016

@author: Administrator
"""

import numpy as np
from HSIDataLoad import *

n_components = 3
#X_train,Y_train,idx_train,X_test,Y_test,idx_test,data_source=dataLoad3(r"D:\OneDrive\codes\python\RandomForest\data")#另一数据集
#X_train = X_train.transpose()
#X_test = X_test.transpose()

#from keras.utils import np_utils
test_size = 0.9
X_data,Y_data,data_source,idx_data=datasetLoad2("D:/data/HSI")
#X_data,data_source = PCA_data_Source(data_source,idx_data,n_components=n_components)
Y_data=np_utils.categorical_probas_to_classes(Y_data)
X_train,X_test,Y_train,Y_test,idx_train,idx_test=datasetSplit(X_data,Y_data,idx_data,num_calss=16,test_size=test_size)
Y_train = np_utils.categorical_probas_to_classes(Y_train)+1
Y_test = np_utils.categorical_probas_to_classes(Y_test)+1


#%% 模型初始化
"""
alpha:L2 regularzation
beta:only used in ADAM
learning_rate : {‘constant’, ‘invscaling’, ‘adaptive’}, 学习率，在sgd方法中改变
momentum：动量，在sgd方法中使用
activation : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
"""
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
clf=MLPClassifier(solver="lbfgs",alpha=1E-5,hidden_layer_sizes=(200,64,16),max_iter=5000)
clf.fit(X_train,Y_train)

Y_pred=clf.predict(X_test)
report =classification_report(Y_pred,Y_test)
acu=accuracy_score(Y_pred,Y_test)
