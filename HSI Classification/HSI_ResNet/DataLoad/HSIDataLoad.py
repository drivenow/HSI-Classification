# -*- coding: utf-8 -*-
import numpy as np
from keras.utils import np_utils
from sklearn.cross_validation import StratifiedShuffleSplit

#%% 
"""
X_train,Y_train,idx_train,X_test,Y_test,idx_test,data_source=dataLoad3(rootPath)
data_source: 21025*200, 有训练集和测试集的index编号。
训练集和测试集直接从文件中读取
"""

def dataLoad3(rootPath):
#    rootPath = r'G:\OneDrive\codes\python\RandomForest\data'
#    rootPath = r'D:\OneDrive\codes\python\RandomForest\data'
#    trainPath = rootPath+r'\train_binal'
#    testPath = rootPath+r'\test_binal'
#    imgPath = rootPath+r'\img_binal'
    trainPath = rootPath+r'\train'
    testPath = rootPath+r'\test'
    
    trainlabPath = rootPath+r'\trainlab'
    trainidxPath = rootPath+r'\trainidx'
    testlabPath = rootPath+r'\testlab'
    testidxPath = rootPath+r'\testidx'
    imgPath = rootPath+r'\img'
      
    X_train = np.loadtxt(open(trainPath,"rb"),delimiter=",",skiprows=0,dtype=np.float)
    Y_train = np.loadtxt(open(trainlabPath,"rb"),delimiter=",",skiprows=0,dtype=np.int)
    idx_train = np.loadtxt(open(trainidxPath,"rb"),delimiter=",",skiprows=0,dtype=np.int)
    X_test = np.loadtxt(open(testPath,"rb"),delimiter=",",skiprows=0,dtype=np.float)
    Y_test = np.loadtxt(open(testlabPath,"rb"),delimiter=",",skiprows=0,dtype=np.int)
    idx_test = np.loadtxt(open(testidxPath,"rb"),delimiter=",",skiprows=0,dtype=np.int)
    img = np.loadtxt(open(imgPath,"rb"),delimiter=',',skiprows=0,dtype=np.float)
    data_source = img.transpose()
    
    return X_train,Y_train,idx_train,X_test,Y_test,idx_test,data_source
    
#%% dataset2
"""
X_data,Y_data,data_source,idx_data=datasetLoad2(rootPath)#未划分训练集测试集的数据(不包括背景点)
Y_data=np_utils.categorical_probas_to_classes(Y_data)
X_train,X_test,Y_train,Y_test,idx_train,idx_test=datasetSplit(X_data,Y_data,idx_data,num_calss=16,test_size=tes_size)
data_source: 21025*200, 有训练集和测试集的index编号
手动分割训练集和测试集
"""

def datasetSplit(data,lab,idx_of_data,num_calss,test_size):
    ssp = StratifiedShuffleSplit(lab,n_iter=1,test_size=test_size)
    for trainlab,testlab in ssp:
        print("train:\n%s\ntest:\n%s" % (trainlab,testlab))
    X_train=data[trainlab]
    X_test=data[testlab]
    Y_train=np_utils.to_categorical(lab[trainlab],num_calss)
    Y_test=np_utils.to_categorical(lab[testlab],num_calss)
    idx_train=idx_of_data[trainlab]
    idx_test=idx_of_data[testlab]
    return X_train,X_test,Y_train,Y_test,idx_train,idx_test


def datasetLoad2(rootPath):
#    rootPath = r'D:/data/HSI'
    Xpath=rootPath+'/labeled_data.2.28.txt'
    Ypath=rootPath+'/data_label.2.28.txt'
    imgPath=rootPath+'/data_source.2.28.txt'
    idxPath=rootPath+'/labeled_idx.2.28.txt'
    
    X_data = np.loadtxt(open(Xpath,"rb"),delimiter=",",skiprows=0,dtype=np.float)
    Y_data = np.loadtxt(open(Ypath,"rb"),delimiter=",",skiprows=0,dtype=np.int)
    Y_data=np_utils.to_categorical(Y_data-1,16)
    data_source=np.loadtxt(open(imgPath,"rb"),delimiter=",",skiprows=0,dtype=np.float)
    idx_data=np.loadtxt(open(idxPath,"rb"),delimiter=",",skiprows=0,dtype=np.int)
    idx_data=idx_data-1
    
    return X_data,Y_data,data_source,idx_data
    
#%%　将数据由１Ｄ索引变为２Ｄ
    
#%% 
"""
# X_data,Y_data,data_source,idx_data=datasetLoad1(rootPath,block_size=7)
Y_data=np_utils.categorical_probas_to_classes(Y_data)
X_train,X_test,Y_train,Y_test,idx_train,idx_test=datasetSplit(X_data,Y_data,idx_data,num_calss=16,test_size=0.9)
# data_source: 21025*200, 有训练集和测试集的index编号
# 返回：n*200*3*3 的高光谱数据 和标签
"""

#1d索引转换为2d
def indexTransform2D(index_1d):
    xidx = index_1d/145
    yidx = index_1d%145
    return int(xidx),int(yidx)

#取数据的领域范围，若超出边界，以边界点来代替
def neighbourhood(idx,block_size):
    xidx,yidx = indexTransform2D(idx)
    x_neighbourhood = [] #x的领域范围
    y_neighbourhood = []
    x_border_left = xidx-int(block_size/2)
    x_border_right = xidx+int(block_size/2)
    y_border_left = yidx-int(block_size/2)
    y_border_right = yidx+int(block_size/2)
    if x_border_left<0:
        x_neighbourhood.extend(abs(x_border_left)*[0])#超出边界，以边界点来代替
        x_neighbourhood.extend(range(0,xidx))
    else:
        x_neighbourhood.extend(range(x_border_left,xidx))
        
    if x_border_right>144:
        x_neighbourhood.extend(range(xidx,145))
        x_neighbourhood.extend(abs(x_border_right-144)*[144])   
    else:
        x_neighbourhood.extend(range(xidx,x_border_right+1))
        
    if y_border_left<0:
        y_neighbourhood.extend(abs(y_border_left)*[0])#超出边界，以边界点来代替
        y_neighbourhood.extend(range(0,yidx))
    else:
        y_neighbourhood.extend(range(y_border_left,yidx))
        
    if y_border_right>144:
        y_neighbourhood.extend(range(yidx,145))
        y_neighbourhood.extend(abs(y_border_right-144)*[144])   
    else:
        y_neighbourhood.extend(range(yidx,y_border_right+1))
    return x_neighbourhood,y_neighbourhood
        

"""
idx_data：有标签的数据的1D索引
data_source : 3D数据块
block_size: 选取像素点周围邻域，组成一个更大的数据块,block_size取奇数
"""   
def blockTansform(idx_data,data_source,block_size):
    samples = len(idx_data)
    X_data = np.zeros((samples,block_size,block_size,200))
    for ii,idx in enumerate(idx_data):
        x_range,y_range = neighbourhood(idx,block_size)#求邻域数据的行列范围
        for iidx,i in enumerate(x_range):
            for jidx,j in enumerate(y_range):
                X_data[ii,iidx,jidx,:] = data_source[i,j,:]
    return X_data

def datasetLoad1(rootPath,block_size):
#    rootPath = r'D:/data/HSI'
    Ypath=rootPath+'/data_label.2.28.txt'
    imgPath=rootPath+'/data_source.2.28.txt'
    idxPath=rootPath+'/labeled_idx.2.28.txt'
    
    
    Y_data = np.loadtxt(open(Ypath,"rb"),delimiter=",",skiprows=0,dtype=np.int)
    Y_data=np_utils.to_categorical(Y_data-1,16)
    
    data_source=np.loadtxt(open(imgPath,"rb"),delimiter=",",skiprows=0,dtype=np.float)
    new_data_source = np.zeros((145,145,200))
    for i in range(200):
        new_data_source[:,:,i] = data_source[:,i].reshape(145,145)
    
    idx_data=np.loadtxt(open(idxPath,"rb"),delimiter=",",skiprows=0,dtype=np.int)
    idx_data=idx_data-1
    
    X_data = blockTansform(idx_data,new_data_source,block_size)
    
    return X_data,Y_data,new_data_source,idx_data
    
#%% PCA提取
def PCA_data_Source(data_source, idx_data, n_components):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components,svd_solver="full")
    new_data_source = pca.fit(data_source).transform(data_source)
    X_data = new_data_source[idx_data,:]
    
    return X_data


#%% 图像增大数据量
def shift(X_sample, padding):
    a = np.zeros((padding))#两边补0的长度
    tmp = np.concatenate((a,X_sample))
    tmp = np.concatenate((tmp,a))
    start_window = np.random.randint(padding*2)
    return tmp[start_window:start_window+200]

def messedup(X_sample, mess_window):
    import copy 
    tmp = copy.copy(X_sample)
    dim = X_sample.size
    start_window = np.random.randint((dim-mess_window))
    tmp[start_window:start_window+10] = 0
    return tmp
    
def data_Augmentation(X_data,Y_data,padding,mess_window):
    sample,dim = X_data.shape
    sample,lab = Y_data.shape
    new_X_data = np.zeros((sample*2,dim))
    new_Y_data = np.zeros((sample*2,lab))
    
    for i in range(sample):
        new_X_data[2*i,:] = shift(X_data[i,:], padding)
        new_X_data[2*i+1,:] = messedup(X_data[i,:], mess_window)
        new_Y_data[2*i,:] = Y_data[i,:]
        new_Y_data[2*i+1,:] = Y_data[i,:]
        
    return new_X_data,new_Y_data

    

