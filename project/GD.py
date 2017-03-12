# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 21:43:14 2015
@author: shenjunling
“””
本算法实现线性回归实验。
输入：年份（2000-2013），年份对应的房价
输出：年份—房价成线性相关的预测直线，损失函数的变化图
参数说明：
X : 年份，同减去2000以标准化
Y : 房价
Alpha ：梯度下降法的步长
Theta ：权重参数
Maxiter : 最大迭代次数
“””
"""

#%%
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
x=np.linspace(2000,2013,14)
x=np.subtract(x,[2000]*14)
#x = np.divide(np.subtract(x,min(x)),(max(x)-min(x)))
#x = np.divide(np.subtract(x,x.mean()),x.var())
y=[2,2.500,2.900,3.147,4.515,4.903,5.365,5.704,6.853,
7.971,8.561,10.000,11.280,12.900]
y = np.array(y)
#y = np.subtract(y,y.mean())
#y = np.divide(np.subtract(y,y.mean()),y.var())
#y = np.divide(np.subtract(y,min(y)),(max(y)-min(y)))
#绘制原始数据散点图
plt.figure(1)
plt.scatter(x,y)

#%%
def regression(x,theta):
    return x*theta[1]+theta[0]
#初始参数
alpha=0.0001
theta0=[1,1]
theta=theta0
maxiter = 50
iterator=0
while True:
    #迭代次数
    iterator=iterator+1
    #预测数据    
    y0=regression(x,theta)
    #误差
    error=sum((y0-y)**2)
    #绘制误差曲线
    if(iterator%1==0):
        plt.figure(2)
        plt.scatter(iterator,error)
    #梯度
 #   gradient=sum(np.multiply(np.subtract(y0,y),x))
    gradient=sum((y0-y)*x)
    #更新theta
    theta=np.subtract(theta,[alpha*gradient]*2)
    #终止条件
    if error<0.001 or iterator>maxiter:
        break
plt.figure(2)
plt.xlabel('iter')
plt.ylabel('loss')
plt.show(False)
#绘制回归曲线
plt.figure(1)
plt.plot(x,y0)
plt.xlabel('year')
plt.ylabel('price')
plt.show(False)
