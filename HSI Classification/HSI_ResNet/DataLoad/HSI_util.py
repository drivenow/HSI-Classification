# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 17:29:43 2017

@author: Administrator
"""
import numpy as np
import scipy.io as sio  

rootPath = "G:/data/HSI"


def reshape_2dat(dat):
    rows,cols,bands = dat.shape
    img=np.zeros((bands,rows*cols))
    for i in range(bands):
        x=dat[:,:,i]
        x=np.reshape(x,(1,rows*cols))
        img[i,:]=x
    return img,rows,cols,bands
    
def linear(X,a,b):
    X=X.flatten()
    x=np.sort(X)
    L=x.shape[0]
    lmin=x[max(int(np.ceil(L*a)),1)]
    lmax=x[min(int(np.floor(L*b)),L-1)]
    return lmax,lmin
    
def line_dat(dat,rdown,rup):
    img=dat
    lmax,lmin = linear(dat, rdown, rup)
    img[dat<lmin] = lmin
    img[dat>lmax] = lmax#去掉顶端和末端
    img = (img-lmin) / lmax#归一化
    return img

"""
#img,labeled_data,data_label,labeled_idx,rows,cols,bands,ctgs = load_mat(rootPath, flag="Indian")
#img,labeled_data,data_label,labeled_idx,rows,cols,bands,ctgs = load_mat(rootPath, flag="Pavia") 
"""    
def load_mat(rootPath, flag="Indian"): 
    if flag=="Indian":
        data=sio.loadmat(rootPath+"/Indian_pines_corrected.mat")
        label=sio.loadmat(rootPath+"/Indian_gt.mat")
        dat=data['indian_pines_corrected']
        lab=label['indian_pines_gt']
        img,rows,cols,bands=reshape_2dat(dat)
        img = np.transpose(img)
    elif flag=="Pavia":
        data=sio.loadmat(rootPath+"/PaviaU.mat")
        label=sio.loadmat(rootPath+"/PaviaU_gt.mat")
        dat=data['paviaU']
        lab=label['paviaU_gt']
        img,rows,cols,bands=reshape_2dat(dat)
        img = np.transpose(img)
    rdown = 0.001
    rup = 0.999
    img = line_dat(img, rdown, rup)
    img_gt = lab.flatten()
    ctgs=np.max(lab)
    
    labeled_idx = []
    labeled_data = []
    data_label = []
    for idx,i in enumerate(img_gt):
        if i!=0:
            labeled_idx.append(idx)
            labeled_data.append(img[int(idx),:])
            data_label.append(i)
    labeled_data = np.array(labeled_data)
    data_label = np.array(data_label)
    labeled_idx = np.array(labeled_idx)
    
    return img,labeled_data,data_label,labeled_idx,rows,cols,bands,ctgs

        