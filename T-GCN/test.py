# -*- coding: utf-8 -*-
import pickle as pkl
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from input_data import *
from tgcn import tgcnCell
#from gru import GRUCell 

from visualization import plot_result,plot_error
from sklearn.metrics import mean_squared_error,mean_absolute_error
#import matplotlib.pyplot as plt
import time

time_start = time.time()
###### Settings ######
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('training_epoch', 10000, 'Number of epochs to train.')
flags.DEFINE_integer('gru_units', 64, 'hidden units of gru.')
flags.DEFINE_integer('seq_len',12 , '  time length of inputs.')
flags.DEFINE_integer('pre_len', 6, 'time length of prediction.')
flags.DEFINE_float('train_rate', 0.95, 'rate of training set.')
flags.DEFINE_integer('batch_size', 32, 'batch size.')
flags.DEFINE_string('dataset', 'cal', 'sz or los.')
flags.DEFINE_string('model_name', 'tgcn', 'tgcn')
model_name = FLAGS.model_name
data_name = FLAGS.dataset
train_rate =  FLAGS.train_rate
seq_len = FLAGS.seq_len
output_dim = pre_len = FLAGS.pre_len
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
gru_units = FLAGS.gru_units

###### load data ######

if data_name == 'cal':
    train_data, adj1 = load_cal_data('cal')
    
data = load_test_data('cal')

# ###### adj normalize
# adj1 = np.mat(adj,dtype=np.float32)
thres = 10000
adj = adj1
print(adj)
adj[adj>thres]=0
adj = adj/np.max(adj)
for i in range(adj.shape[0]):
    for j in range(adj.shape[0]):
        if i == j:
            adj[i,j] = 1
        else:
            if adj[i,j]>0:
                adj[i,j] = 1-adj[i,j]
    
print(adj)
# adj = adj1

num_nodes = data.shape[2]
data1 = data
print(data1.shape)
#### normalization
max_value = np.max(np.mat(train_data,dtype=np.float32))
data1  = data1/max_value
testX = data1

def TGCN(_X, _weights, _biases):
    ###
    cell_1 = tgcnCell(gru_units, adj, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
    m = []
    for i in outputs:
        o = tf.reshape(i,shape=[-1,num_nodes,gru_units])
        o = tf.reshape(o,shape=[-1,gru_units])
        m.append(o)
    last_output = m[-1]
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    output = tf.reshape(output,shape=[-1,num_nodes,1])
    output = tf.transpose(output, perm=[0,2,1])
    output = tf.reshape(output, shape=[-1,num_nodes])
    return output, m, states
        
###### placeholders ######
inputs = tf.placeholder(tf.float32, shape=[None, seq_len, num_nodes])
labels = tf.placeholder(tf.float32, shape=[None, 1, num_nodes])

# Graph weights
weights = {
    'out': tf.Variable(tf.random_normal([gru_units, 1], mean=1.0), name='weight_o')}
biases = {
    'out': tf.Variable(tf.random_normal([1]),name='bias_o')}

if model_name == 'tgcn':
    pred,ttts,ttto = TGCN(inputs, weights, biases)

y_pred = pred

###### Initialize session ######
variables = tf.global_variables()
saver = tf.train.Saver(tf.global_variables())  
#sess = tf.Session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

out = 'out/%s'%(model_name)
#out = 'out/%s_%s'%(model_name,'perturbation')
path1 = '%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r'%(model_name,data_name,lr,batch_size,gru_units,seq_len,pre_len,training_epoch)
path = os.path.join(out,path1)
if not os.path.exists(path):
    os.makedirs(path)
    
# saver.restore(sess, 'out/tgcn/pre3_tgcn_lr0.0001_batch64_unit64_seq12_epoch10000/model_100/TGCN_pre_0-0')
saver.restore(sess, 'out/tgcn/pre3_vs_lr0.0001_epoch10000_/model_100/TGCN_pre_0-0')
test_output15 = sess.run(y_pred, feed_dict = {inputs:testX})
print(test_output15[0][0],len(test_output15),len(test_output15[0]))
test_output15 = np.reshape(test_output15,[-1,num_nodes])
test_output15 = test_output15*max_value
print(test_output15[0][0],test_output15.shape)

saver.restore(sess, 'out/tgcn/pre6_vs_lr0.0001_epoch10000/model_100/TGCN_pre_200-200')

test_output30 = sess.run(y_pred, feed_dict = {inputs:testX})
print(test_output30[0][0],len(test_output30),len(test_output30[0]))
test_output30 = np.reshape(test_output30,[-1,num_nodes])
test_output30 = test_output30*max_value
print(test_output30[0][0],test_output30.shape)

saver.restore(sess, 'out/tgcn/pre9_tgcn_lr0.0001_batch64_unit64_seq12_epoch10000_stdnorm/model_100/TGCN_pre_100-100')

test_output45 = sess.run(y_pred, feed_dict = {inputs:testX})
print(test_output45[0][0],len(test_output45),len(test_output45[0]))
test_output45 = np.reshape(test_output45,[-1,num_nodes])
test_output45 = test_output45*max_value
print(test_output45[0][0],test_output45.shape)

test_csv = np.zeros((80*3*228),dtype=np.float32)
for i in range(80):
    test_csv[(i*3*228):(i*3*228+228)]=test_output15[i,:]
    test_csv[(i*3*228+228):(i*3*228+2*228)]=test_output30[i,:]
    test_csv[(i*3*228+2*228):(i*3*228+3*228)]=test_output45[i,:]

var = pd.DataFrame(test_csv)
var.to_csv('./test_result_total.csv',index = False,header = False)

time_end = time.time()
print(time_end-time_start,'s')
