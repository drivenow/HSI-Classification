# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:18:59 2017

@author: Administrator
"""

from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np

base_path = "result/13 98.6/"

def plot_result(lab,index,imname):
    im = np.zeros((145,145))+16
    for idx,i in enumerate(index):
        row = int(np.floor(i/145))
        col = int(i%145)
        im[row][col] = lab[idx]
    
    fig = plt.figure(figsize=(6,6))
#    cmap = plt.get_cmap('tab20', 20)
    color=["red","green","lightcoral","red","chocolate","yellow","orange","slategray","indigo",
            "blue","teal","skyblue","darkorchid","gold","silver","darkgreen","black"]
    cmap=colors.ListedColormap(color)
    sc = plt.imshow(im, cmap = cmap)
#    plt.colorbar(sc, ticks = np.arange(0,17))
    axes=plt.subplot(111)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.savefig(base_path+imname+".jpg")
#    plt.imsave(base_path+imname+".jpg",im, cmap = cmap)
    
def plot_train(total_lab,total_index,train_index, imname):
    im = np.zeros((145,145))+16
    for idx,i in enumerate(total_index):
        row = int(np.floor(i/145))
        col = int(i%145)
        im[row][col] = total_lab[idx]
    for idx,i in enumerate(train_index):
        row = int(np.floor(i/145))
        col = int(i%145)
        im[row][col] = 16
    
    fig = plt.figure(figsize=(6,6))
    color=["red","green","lightcoral","red","chocolate","yellow","orange","slategray","indigo",
            "blue","teal","skyblue","darkorchid","gold","silver","darkgreen","black"]
    cmap=colors.ListedColormap(color)
    sc = plt.imshow(im, cmap = cmap)
    plt.colorbar(sc, ticks = np.arange(0,17),fraction=0.046, pad=0.04)
    axes=plt.subplot(111)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.savefig(base_path+imname+".jpg")

    
        
Y_data = np.loadtxt(base_path+"Y_data.txt")
idx_train = np.loadtxt(base_path+"idx_train.txt")
idx_test = np.loadtxt(base_path+"idx_test.txt")
idx_data = np.loadtxt(base_path+"idx_data.txt")
pred = np.loadtxt(base_path+"pred.txt")
train_lab=np.loadtxt(base_path+"train_lab.txt")

pred1 = np.zeros(pred.shape[0])+16
for i in range(pred.shape[0]):
    pred1[i] = np.argmax(pred[i])

idx_data_list = list(idx_data)
accu_count={}
accu_total = {}
for i in range(16):
    accu_count[i]=0
    accu_total[i]=0

for idx,test in enumerate(idx_test):
    place = idx_data_list.index(test)
    accu_total[Y_data[place]]=accu_total[Y_data[place]]+1
    if(Y_data[place]==pred1[idx]):
        accu_count[Y_data[place]] = accu_count[Y_data[place]]+1

accu = []#class-wise accu
for i in range(16):
    accu.append(accu_count[i]/accu_total[i])
total_accu = sum(accu_count.values())/sum(accu_total.values())#accuracy


#plot_result(Y_data, idx_data, "groud_truth")#
#plot_train(Y_data,idx_data,idx_train,"train_sample")
lab_pred = np.concatenate((pred1,train_lab))
idx_pred = np.concatenate((idx_test,idx_train))
plot_result(lab_pred, idx_pred,"test_sample")#

