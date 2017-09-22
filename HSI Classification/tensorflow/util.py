# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:06:19 2017

@author: Shenjunling
"""

import tensorflow as tf
import numpy as np

"""
kernel_shape:[kernel_size,kenel_size,in_dim,out_dim]
""" 
def convolution2d(x, kernel_shape, name):
    with tf.variable_scope(name):
        kernel = tf.Variable(tf.truncated_normal(kernel_shape,
                              dtype=tf.float32,stddev=0.1))
        conv = tf.nn.conv2d(x, kernel, [1,1,1,1], padding="SAME")
        return conv
    
def batchNorm(inputs, is_train, is_conv_out=True, name="bn"):
    with tf.variable_scope(name):
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
        

        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])   

        train_mean = tf.assign(pop_mean,
                               batch_mean)
        train_var = tf.assign(pop_var,
                              batch_var)#nn,moments返回的是tensor,但是control_dependencies必须要是op
        with tf.control_dependencies([train_mean, train_var]):
            mean, var = tf.cond(is_train, lambda: (batch_mean, batch_var),
                            lambda: (pop_mean, pop_var))#tensorflow 的 if
            return tf.nn.batch_normalization(inputs,
                mean, var, beta, scale, 0.001)
  

    
    
"""
w = tf.get_variable('weights', [x.get_shape()[1], out_dim],
                            tf.float32, initializer=tf.random_normal_initializer(
                                stddev=np.sqrt(1.0/out_dim)))
"""    
def dense(x, w_shape, activation, name):
    with tf.variable_scope(name, reuse=False):
        w = tf.get_variable("weight",w_shape, tf.float32,initializer=
                            tf.random_normal_initializer(stddev=np.sqrt(1.0/w_shape[1])))
        b= tf.get_variable("bias",[w_shape[1]],tf.float32,initializer=tf.constant_initializer(0.0))
            
        if activation=="relu":
            if w not in tf.get_collection("fc_weights"):
                tf.add_to_collection("fc_weights",w)
            return tf.nn.relu(tf.matmul(x,w)+b)
        elif activation=="softmax":
            return tf.nn.softmax(tf.matmul(x,w)+b)
    

def conv_block(x,nb_filter,is_train, kernel_size=3,name="unit"):
    with tf.variable_scope(name):
        input_featruemap = int(x.get_shape()[3])
        k1,k2,k3 = nb_filter
        input_shape1 = [1,1,input_featruemap,k1]
        input_shape2 = [kernel_size,kernel_size,k1,k2]
        input_shape3 = [1,1,k2,k3]
        input_shape4 = [1,1,input_featruemap,k3]
        
        out = convolution2d(x, input_shape1, "conv1")
        out = batchNorm(out, is_train, True)
        out = tf.nn.relu(out,"relu1")
        
        out = convolution2d(out, input_shape2,"conv2")
        out = batchNorm(out, is_train, True)
        out = tf.nn.relu(out, "relu2")
        
        out = convolution2d(out, input_shape3, "conv3")
        out = batchNorm(out, is_train, True)
        
        x = convolution2d(x, input_shape4, "conv4")
        out = batchNorm(out, is_train, True)
        
        out = tf.add(out,x)
        out = tf.nn.relu(out, name = "relu3")
    return out
    
def identity_block(x, nb_filter, is_train, kernel_size=3, name="unit"):
    with tf.variable_scope(name):
        input_featruemap = int(x.get_shape()[3])
        k1,k2,k3 = nb_filter
        input_shape1 = [1,1,input_featruemap,k1]
        input_shape2 = [kernel_size,kernel_size,k1,k2]
        input_shape3 = [1,1,k2,k3]
        
        out = convolution2d(x, input_shape1, "conv1")
        out = batchNorm(out, is_train, True)
        out = tf.nn.relu(out, "relu1")
        
        out = convolution2d(out, input_shape2, "conv2")
        out = batchNorm(out, is_train, True)
        out = tf.nn.relu(out, "relu2")
        
        out = convolution2d(out, input_shape3, "conv3")
        out = batchNorm(out, is_train, True)
        
        out = tf.add(out,x)
        out = tf.nn.relu(out, "relu3")
    return out