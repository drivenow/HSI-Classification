# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 08:54:18 2017

@author: fullmetal
"""
import numpy as np

def get_accu(f):
    result = []
    flag =True
    for line in f:
        if(flag==True):
            result.append(float(line))
            flag=False
        if(line.startswith("=")):
            flag=True
    return result

base_path="result/pca30/"
block_size=[3,5,7,9,11,13,15,17]
block_accu = []

for size in block_size:
    f = open(base_path+"result"+str(size)+"_res1_30.txt")
    result = get_accu(f)
    block_accu.append(np.mean(result))
    print(np.mean(result))
    
result = get_accu(open(base_path+"result17_res1_30_85.txt"))
print(np.mean(result))
    

    