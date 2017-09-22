# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 21:29:07 2017

@author: Administrator
"""
import numpy as np
from keras.utils import np_utils
from sklearn.cross_validation import StratifiedShuffleSplit
from HSI_util import load_mat


"""
HSI = HSIData(rootPath,"Indian")
X_data = HSI.X_data     #(10336,200)
Y_data = HSI.Y_data    #(10336,16)
data_source = HSI.data_source    #(21025,200)
idx_data = HSI.idx_data   #(10336,)
"""
class HSIData:
#    def __init__(self,rootPath):
#        Xpath=rootPath+'/labeled_data.2.28.txt'
#        Ypath=rootPath+'/data_label.2.28.txt'
#        imgPath=rootPath+'/data_source.2.28.txt'
#        idxPath=rootPath+'/labeled_idx.2.28.txt'
#        
#        self.X_data = np.loadtxt(open(Xpath,"rb"),delimiter=",",skiprows=0,dtype=np.float)
#        Y_data = np.loadtxt(open(Ypath,"rb"),delimiter=",",skiprows=0,dtype=np.int)
#        self.Y_data=np_utils.to_categorical(Y_data-1,16)
#        self.data_source=np.loadtxt(open(imgPath,"rb"),delimiter=",",skiprows=0,dtype=np.float)
#        idx_data=np.loadtxt(open(idxPath,"rb"),delimiter=",",skiprows=0,dtype=np.int)
#        self.idx_data=idx_data-1
#        self.rows =145
#        self.cols =145
#        self.ctgs=16
    """
    两个数据集加载
    """
    def __init__(self, rootPath, flag="Indian"):
        img,labeled_data,data_label,labeled_idx,rows,cols,bands,ctgs = load_mat(rootPath, flag)
#        img,labeled_data,data_label,labeled_idx,rows,cols,bands,ctgs = load_mat(rootPath, flag="Pavia")
        self.X_data=labeled_data
        self.Y_data=np_utils.to_categorical(data_label-1,ctgs)
        self.idx_data=labeled_idx
        self.data_source=img
        self.rows = rows
        self.cols=cols
        self.bands=bands
        self.ctgs=ctgs
        
        
    """
    X_data,Y_data,data_source,idx_data=datasetLoad2(rootPath)#未划分训练集测试集的数据(不包括背景点)
    Y_data=np_utils.categorical_probas_to_classes(Y_data)
    X_train,X_test,Y_train,Y_test,idx_train,idx_test=datasetSplit(X_data,Y_data,idx_data,num_calss=16,test_size=tes_size)
    data_source: 21025*200, 有训练集和测试集的index编号
    手动分割训练集和测试集
    """
    def datasetSplit(self,data,lab,idx_of_data,num_calss,test_size):
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
    
    #%% PCA提取主成分
    def PCA_data_Source(self,data_source, n_components):
        from sklearn.decomposition import PCA
        pca = PCA(n_components = n_components,svd_solver="full")
        new_data_source = pca.fit(data_source).transform(data_source)
        
        return new_data_source
    
    
    #%% 图像增大数据量
    def shift(self,X_sample, padding):
        a = np.zeros((padding))#两边补0的长度
        tmp = np.concatenate((a,X_sample))
        tmp = np.concatenate((tmp,a))
        start_window = np.random.randint(padding*2)
        return tmp[start_window:start_window+200]
    
    def messedup(self,X_sample, mess_window):
        import copy 
        tmp = copy.copy(X_sample)
        dim = X_sample.size
        start_window = np.random.randint((dim-mess_window))
        tmp[start_window:start_window+10] = 0
        return tmp
        
    def data_Augmentation(self,X_data,Y_data,padding,mess_window):
        sample,dim = X_data.shape
        sample,lab = Y_data.shape
        new_X_data = np.zeros((sample*2,dim))
        new_Y_data = np.zeros((sample*2,lab))
        
        for i in range(self,sample):
            new_X_data[2*i,:] = self.shift(X_data[i,:], padding)
            new_X_data[2*i+1,:] = self.messedup(X_data[i,:], mess_window)
            new_Y_data[2*i,:] = Y_data[i,:]
            new_Y_data[2*i+1,:] = Y_data[i,:]
        return new_X_data,new_Y_data
    #%% 加载邻域数据   
    """
    X_data = HSI.getNeighborData(data_source=data_source,idx_data=idx_data,block_size=block_size)
    Y_data = np_utils.categorical_probas_to_classes(Y_data)
    X_train,X_test,Y_train,Y_test,idx_train,idx_test = HSI.datasetSplit(X_data,Y_data,idx_data,16,test_size = test_size)
    # data_source: 21025*200, 有训练集和测试集的index编号
    # 返回：n*200*k*k 的高光谱数据 和标签
    """
    
    #1d索引转换为2d
    def indexTransform2D(self,index_1d):
        xidx = index_1d/self.cols
        if(xidx==0 & index_1d>self.cols):
            xidx=self.rows
        yidx = index_1d%self.cols
        return int(xidx),int(yidx)
    
    #取数据的领域范围，若超出边界，以边界点来代替
    def neighbourhood(self,idx,block_size):
        xidx,yidx = self.indexTransform2D(idx)
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
            
        if x_border_right>self.rows-1:
            x_neighbourhood.extend(range(xidx,self.rows))
            x_neighbourhood.extend(abs(x_border_right-self.rows+1)*[self.rows-1])   
        else:
            x_neighbourhood.extend(range(xidx,x_border_right+1))
            
        if y_border_left<0:
            y_neighbourhood.extend(abs(y_border_left)*[0])#超出边界，以边界点来代替
            y_neighbourhood.extend(range(0,yidx))
        else:
            y_neighbourhood.extend(range(y_border_left,yidx))
            
        if y_border_right>self.cols-1:
            y_neighbourhood.extend(range(yidx,self.cols))
            y_neighbourhood.extend(abs(y_border_right-self.cols+1)*[self.cols-1])   
        else:
            y_neighbourhood.extend(range(yidx,y_border_right+1))
        return x_neighbourhood,y_neighbourhood
            
    
    """
    idx_data：有标签的数据的1D索引
    data_source : 3D数据块
    block_size: 选取像素点周围邻域，组成一个更大的数据块,block_size取奇数
    """   
    def blockTansform(self,idx_data,data_source,block_size):
        samples = len(idx_data)
        bands = data_source.shape[2]
        X_data = np.zeros((samples,block_size,block_size,bands))
        for ii,idx in enumerate(idx_data):
            x_range,y_range = self.neighbourhood(idx,block_size)#求邻域数据的行列范围
#            print(y_range)
            for iidx,i in enumerate(x_range):
                for jidx,j in enumerate(y_range):
                    X_data[ii,iidx,jidx,:] = data_source[i,j,:]
        return X_data

    
    def getNeighborData(self,data_source,idx_data,block_size):
        bands = data_source.shape[1]
        new_data_source = np.zeros((self.rows,self.cols,bands))
        for i in range(bands):
            new_data_source[:,:,i] = data_source[:,i].reshape(self.rows,self.cols)
        
        X_data = self.blockTansform(idx_data,new_data_source,block_size)
        return X_data