#coding=utf-8
import numpy as np
import tensorflow as tf

# Define parameters
FLAGS = tf.app.flags.FLAGS

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("issync", 1, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")
tf.app.flags.DEFINE_string("mode","train","train|inference")

# Hyperparameters
logdir = ""
nb_epoch = 600
issync = FLAGS.issync
learning_rate = 0.001

#%% data

#logBasePath = "D:/data/mylog/KerasDL/"
#rootPath = r'D:/data/HSI'

rootPath = "G:/data/HSI"
logBasePath = "G:/data/mylog/KerasDL/CNN2d_pca_model"

block_size = 11
test_size = 0.9
#validate_size = 0.8
nb_epoch = 1000
nb_classes = 16
batch_size = 32
l2_lr = 0.1

#是否使用pca
use_pca = True
n_components = 30
if use_pca ==True:
    input_shape = (block_size,block_size,n_components)
else:
    input_shape = (block_size,block_size,200)

#%% data
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

#%%  main
from util import convolution2d,batchNorm,dense,conv_block,identity_block

def main(_):
    #regist ps,worker
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)
  
  
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % FLAGS.task_index,
                    cluster=cluster)):
            global_step = tf.Variable(0, name='global_step', trainable=False)
          #******************1.build_model here***********************
          #tf graph input   
            img = tf.placeholder(tf.float32, shape=(None,block_size,block_size,n_components),name="image")
            label = tf.placeholder(tf.float32, [None, nb_classes])
            keep_prob = tf.placeholder(tf.float32)
            is_train = tf.placeholder(tf.bool)
            
            #1.build model
            res1 = conv_block(img,[64,64,256], is_train, 3, "res1")
            res2 = identity_block(res1,[64,64,256],is_train,3,"res2")
            res3 = identity_block(res2,[128,128,256],is_train,3,"res3")
            res4 = identity_block(res3,[128,128,256], is_train, 3,"res4")
            
          
            flat_input_shape = [-1,int(res1.get_shape()[1])*int(res1.get_shape()[2])*int(res1.get_shape()[3])]
            flat = tf.reshape(res4, flat_input_shape)
            
            fc1 = dense(flat, [flat_input_shape[1], 1024], "relu","fc1")
            fc1 = batchNorm(fc1, is_train, False, "bn1")
            fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)
            
            fc2 = dense(fc1, [1024, 1024], "relu","fc2")
            fc2 = batchNorm(fc2, is_train, False, "bn2")
            fc2 = tf.nn.dropout(fc2, keep_prob=keep_prob)
            
            fc3 = dense(fc2, [1024, nb_classes], "softmax","fc3")
          
          
          #*******************2.model optimizer && train_op**************************
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            #2.train_op
            loss_ts = -tf.reduce_sum(label*tf.log(tf.clip_by_value(fc3,1e-10,1.0)))#cross_entropy
            l2_loss = [tf.nn.l2_loss(var) for var in tf.get_collection("fc_weights")]
            l2_loss = tf.multiply(l2_lr, tf.add_n(l2_loss))
            loss_ts = loss_ts+l2_loss
            
            prediction_ts = tf.equal(tf.arg_max(fc3,1), tf.arg_max(label,1))
            accuracy_ts = tf.reduce_mean(tf.cast(prediction_ts, tf.float32))
            
            if issync == 1:
            #同步模式计算更新梯度
                grads_and_vars = optimizer.compute_gradients(loss_ts)
                rep_op = tf.train.SyncReplicasOptimizer(optimizer,
                                                        replicas_to_aggregate=len(worker_hosts),
                                                        replica_id=FLAGS.task_index,
                                                        total_num_replicas=len(worker_hosts),
                                                        use_locking=True)
                train_op = rep_op.apply_gradients(grads_and_vars,
                                               global_step=global_step)
                init_token_op = rep_op.get_init_tokens_op()
                chief_queue_runner = rep_op.get_chief_queue_runner()
            else:
                #异步模式计算更新梯度
                train_op = tf.train.RMSPropOptimizer(0.001).minimize(loss_ts,global_step=global_step)
        
        #******************3.model saver and summary**************************
        accu_sm = tf.summary.scalar("accuracy", accuracy_ts)
        loss_sm = tf.summary.scalar("loss", loss_ts)
        summary_op = tf.summary.merge_all()
        init_op = tf.initialize_all_variables()
        
        
        #%%
        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        if FLAGS.mode == "train":
            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                   logdir=logdir,
                                   init_op=init_op,
                                   summary_op=None,
                                   saver=None,
                                   global_step=global_step,
                                   stop_grace_secs=300,
                                   save_model_secs=10)
        else:
            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                   logdir=logdir,
                                   summary_op=summary_op,
                                   saver=None,
                                   global_step=global_step,
                                   stop_grace_secs=300,
                                   save_model_secs=0)
            
        with sv.prepare_or_wait_for_session(server.target) as sess:
        with sv.prepare_or_wait_for_session(server.target) as sess:
            # 如果是同步模式
            if FLAGS.task_index == 0 and issync == 1:
                sv.start_queue_runners(sess, [chief_queue_runner])
                sess.run(init_token_op)
            #*******************4.run op**********************
            summary_writer = tf.summary.FileWriter("G:/data/mylog/KerasDL/HSI_resnet_dist/logs",
                                       sess.graph)
            sess.run(init_op)
            for it in range(nb_epoch):
                print("epoch %d: [=============]" % it, end=" ")
                data_generator = next_batch(batch_size, X_train_nei, Y_train)
                for X,Y in data_generator:
                    train_op.run(feed_dict={img:X,label:Y, keep_prob:0.7, is_train:True})
                    
                loss,acc,train_summary_str = sess.run([loss_ts, accuracy_ts, summary_op], feed_dict={img:X_train_nei, label:Y_train, keep_prob:0.7, is_train:False})
                
                test_generator = next_batch(batch_size, X_test_nei, Y_test)
                pred_list= np.array([])
                loss_total=0
                for X,Y in test_generator:
                    val_loss,pred = sess.run([loss_ts, prediction_ts], feed_dict={img:X, label:Y, keep_prob:1.0, is_train:True})
                    pred_list=np.concatenate((pred_list,pred),axis=0)
                    loss_total +=val_loss
                pred_list[0]=1
                val_accu = np.mean(pred_list)
                
                print(" - loss: %d - acc: %.4f - val_loss: %.4f - val_acc: %.4f" % (loss,acc,val_loss,val_accu))
                
                #summary
                summary_writer.add_summary(train_summary_str, it)
                summary_writer.flush()
        #别忘了关闭监视器
        sv.stop()

if __name__ == "__main__":
  tf.app.run()

