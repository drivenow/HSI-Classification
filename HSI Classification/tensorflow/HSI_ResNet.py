# -*- coding: utf-8 -*-
"""
Created on Fri May 26 20:03:33 2017
@author: Shenjunling
epoch 594: [learning_rate: 1]
 - loss: 0 - acc: 1.0000 - val_loss: 0.0841 - val_acc: 0.9849
epoch 595: [learning_rate: 1]
 - loss: 0 - acc: 1.0000 - val_loss: 3.0241 - val_acc: 0.9838
epoch 596: [learning_rate: 1]
 - loss: 0 - acc: 1.0000 - val_loss: 0.5930 - val_acc: 0.9814
epoch 597: [learning_rate: 1]
 - loss: 0 - acc: 1.0000 - val_loss: 0.0359 - val_acc: 0.9834
epoch 598: [learning_rate: 1]
 - loss: 0 - acc: 1.0000 - val_loss: 0.3521 - val_acc: 0.9817
epoch 599: [learning_rate: 1]
 - loss: 0 - acc: 1.0000 - val_loss: 4.6297 - val_acc: 0.9829
"""

#logBasePath = "D:/data/mylog/KerasDL/"
#rootPath = r'D:/data/HSI'

rootPath = "G:/data/HSI"
logBasePath = "G:/data/mylog/KerasDL/CNN2d_pca_model"

block_size = 21
test_size = 0.9
#validate_size = 0.8
nb_epoch = 600
epoch = 1
nb_classes = 16
batch_size = 40
l2_lr = 0.1



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

def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    
    count=0
    while count<len(data):
        subidx = idx[count: min(count+num,len(data))]
        data_shuffle = [data[i] for i in subidx]
        lables_shuffle = [labels[i] for i in subidx]
        yield (np.asarray(data_shuffle,dtype=np.float32),np.asarray(lables_shuffle,dtype=np.float32))
        count+=num

HSI = HSIData(rootPath)
X_data = HSI.X_data
Y_data = HSI.Y_data
data_source = HSI.data_source
idx_data = HSI.idx_data

#是否使用PCA降维
if use_pca==True:
    data_source = HSI.PCA_data_Source(data_source,n_components=n_components)
    
X_data_nei = HSI.getNeighborData(data_source,idx_data,block_size)

Y_data = np_utils.categorical_probas_to_classes(Y_data)
X_train_nei,X_test_nei,Y_train,Y_test,idx_train,idx_test = HSI.datasetSplit(X_data_nei,Y_data,idx_data,16,test_size = test_size)


#%%
import tensorflow as tf
import numpy as np
from tensorflow.contrib.keras.api.keras.layers import BatchNormalization

def get_scale(shape):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        receptive_field_size = np.prod(shape[:2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size

    s = np.sqrt(2. / (fan_in + fan_out))
    return s
    
    
"""
kernel_shape:[kernel_size,kenel_size,in_dim,out_dim]
""" 
def convolution2d(x, kernel_shape, name):
    with tf.variable_scope(name):
        s = get_scale(kernel_shape)
        kernel = tf.get_variable("kernel",kernel_shape, tf.float32, initializer=
                             tf.random_normal_initializer(stddev=s))
        conv = tf.nn.conv2d(x, kernel, [1,1,1,1], padding="SAME")
        return conv
    
#def batchNorm(inputs, is_train, is_conv_out=True, name="bn"):
#    with tf.variable_scope(name):
#        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
#        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
#        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
#        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
#        
#
#        if is_conv_out:
#            batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
#        else:
#            batch_mean, batch_var = tf.nn.moments(inputs,[0])   
#
#        train_mean = tf.assign(pop_mean,
#                               batch_mean)
#        train_var = tf.assign(pop_var,
#                              batch_var)#nn,moments返回的是tensor,但是control_dependencies必须要是op
#        with tf.control_dependencies([train_mean, train_var]):
#            mean, var = tf.cond(is_train, lambda: (batch_mean, batch_var),
#                            lambda: (pop_mean, pop_var))#tensorflow 的 if
#            return tf.nn.batch_normalization(inputs,
#                mean, var, beta, scale, 0.001)

def batchNorm(inputs, is_train, is_conv_out=True,name="bn", decay = 0.99):
    with tf.variable_scope(name):
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])
        gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
                            initializer=tf.ones_initializer())
        beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
                            initializer=tf.zeros_initializer())
        pop_mean = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
                            initializer=tf.zeros_initializer(), trainable=True)
        pop_var = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
                            initializer=tf.ones_initializer(), trainable=True)
        
           
        train_mean = tf.assign(pop_mean,
                           pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                          pop_var * decay + batch_var * (1 - decay))
        
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, train_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, train_var)
        
        mean,var = tf.cond(is_train, lambda: (batch_mean, batch_var),
                            lambda: (pop_mean, pop_var))

        return tf.nn.batch_normalization(inputs,
                                        mean, var, beta, gamma, 0.01)

    
    
"""
w = tf.get_variable('weights', [x.get_shape()[1], out_dim],
                            tf.float32, initializer=tf.random_normal_initializer(
                                stddev=np.sqrt(1.0/out_dim)))
"""    
def dense(x, w_shape, activation, name):
    with tf.variable_scope(name, reuse=False):
        s = get_scale(w_shape)
        w = tf.get_variable("weight",w_shape, tf.float32,initializer=
                            tf.random_normal_initializer(stddev=s))
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
        out = tf.nn.relu(out,"relu1")
        out = batchNorm(out, is_train, True, "bn1")
    
        out = convolution2d(out, input_shape2,"conv2")
        out = tf.nn.relu(out, "relu2")
        out = batchNorm(out, is_train, True,"bn2")
        
        out = convolution2d(out, input_shape3, "conv3")
        
        
        x = convolution2d(x, input_shape4, "conv4")
        
        out = tf.add(out,x)
        out = tf.nn.relu(out, name = "relu3")
        out = batchNorm(out, is_train, True,"bn3")
    return out
    
def identity_block(x, nb_filter, is_train, kernel_size=3, name="unit"):
    with tf.variable_scope(name):
        input_featruemap = int(x.get_shape()[3])
        k1,k2,k3 = nb_filter
        input_shape1 = [1,1,input_featruemap,k1]
        input_shape2 = [kernel_size,kernel_size,k1,k2]
        input_shape3 = [1,1,k2,k3]
        
        out = convolution2d(x, input_shape1, "conv1")
        out = tf.nn.relu(out, "relu1")
        out = batchNorm(out, is_train, True,"bn1")
        
        out = convolution2d(out, input_shape2, "conv2")
        out = tf.nn.relu(out, "relu2")
        out = batchNorm(out, is_train, True,"bn2")
        
        out = convolution2d(out, input_shape3, "conv3")
        
        
        out = tf.add(out,x)
        out = tf.nn.relu(out, "relu3")
        out = batchNorm(out, is_train, True,"bn3")
        
    return out
    
#tf graph input   
img = tf.placeholder(tf.float32, shape=(None,block_size,block_size,n_components),name="image")
label = tf.placeholder(tf.float32, [None, nb_classes])
keep_prob = tf.placeholder(tf.float32)
is_train = tf.placeholder(tf.bool)

#1.build model
res1 = conv_block(img,[64,64,256], is_train, 3, "res1")
res2 = identity_block(res1,[64,64,256],is_train,3,"res2")
#res3 = identity_block(res2,[128,128,256],is_train,3,"res3")
#res4 = identity_block(res3,[128,128,256], is_train, 3,"res4")

#sliced = tf.slice(res4,[0,5,5,0],[-1,1,1,int(res1.get_shape()[3])])
#flat_input_shape = [-1,1*1*int(res1.get_shape()[3])]
#flat = tf.reshape(sliced, flat_input_shape)

flat_input_shape = [-1,21*21*int(res1.get_shape()[3])]
flat = tf.reshape(res2, flat_input_shape)


fc1 = dense(flat, [flat_input_shape[1], 1024], "relu","fc1")
fc1 = batchNorm(fc1, is_train, False,"bn1")
fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

fc2 = dense(fc1, [1024, 1024], "relu","fc2")
fc2 = batchNorm(fc2, is_train, False, "bn2")
fc2 = tf.nn.dropout(fc2, keep_prob=keep_prob)

fc3 = dense(fc2, [1024, nb_classes], "softmax","fc3")


#2.train_op
loss_op = -tf.reduce_sum(label*tf.log(tf.clip_by_value(fc3,1e-10,1.0)))#cross_entropy
l2_loss = [tf.nn.l2_loss(var) for var in tf.get_collection("fc_weights")]
l2_loss = tf.multiply(l2_lr, tf.add_n(l2_loss))
loss_op = loss_op+l2_loss

#学习率退火
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.001, global_step, decay_steps=40*60, decay_rate=0.4, staircase=True)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
#    optimizer = tf.train.AdadeltaOptimizer()
    train_op = optimizer.minimize(loss_op, global_step=global_step)


prediction_op = tf.equal(tf.arg_max(fc3,1), tf.arg_max(label,1))
accuracy_op = tf.reduce_mean(tf.cast(prediction_op, tf.float32))

#3.summary
#saver_op = tf.train.Saver([w1,w2])
#saver_op.save(sess,"G:/data")

accu_s = tf.summary.scalar("accuracy", accuracy_op)
loss_s = tf.summary.scalar("loss", loss_op)
merged = tf.summary.merge_all()

init_op = tf.global_variables_initializer()


#sess = tf.Session()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter("G:/data/mylog/KerasDL/HSI_resnet_dist/logs",
                                           sess.graph)
    sess.run(init_op)
    for it in range(nb_epoch):
#        print("epoch %d: [learning_rate: %f]" % (it,sess.run(optimizer._learning_rate)), end=" ")
        print("epoch %d: [learning_rate: 1]" % it)
        data_generator = next_batch(batch_size, X_train_nei, Y_train)
        for X,Y in data_generator:
            train_op.run(feed_dict={img:X,label:Y, keep_prob:0.7, is_train:True})
            
        loss,acc,train_summary_str = sess.run([loss_op, accuracy_op, merged], feed_dict={img:X_train_nei, label:Y_train, keep_prob:0.7, is_train:True})
        
        test_generator = next_batch(batch_size, X_test_nei, Y_test)
        pred_list= np.array([])
        loss_total=0
        for X,Y in test_generator:
            val_loss,pred = sess.run([loss_op, prediction_op], feed_dict={img:X, label:Y, keep_prob:1.0, is_train:False})
            pred_list=np.concatenate((pred_list,pred),axis=0)
            loss_total +=val_loss
        pred_list[0]=1
        val_accu = np.mean(pred_list)
        
        print(" - loss: %d - acc: %.4f - val_loss: %.4f - val_acc: %.4f" % (loss,acc,val_loss,val_accu))
        
        #summary
        summary_writer.add_summary(train_summary_str, it)
        summary_writer.flush()
    
