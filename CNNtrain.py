import os
import time
import numpy as np
# # from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_152
# from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_16
# import skimage
# import skimage.io
# import skimage.transform
import tensorflow as tf
from numpy import mat
import matplotlib.pyplot as plt  
import sklearn
import pandas as pd

n_epoch =200
print_freq =2
batch_size =32
lr = 1e-4
k0 = 63 # num of neighbors
pre_len = 9
training_epoch = 5000

x = tf.placeholder(tf.float32, shape=[None,12,k0+1,1])
# 输出
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y_')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(inputx, W):
  return tf.nn.conv2d(inputx, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(inputx):
  return tf.nn.max_pool(inputx, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def max_pool_1x2(inputx):
  return tf.nn.max_pool(inputx, ksize=[1, 1, 2, 1],
                        strides=[1, 1, 2, 1], padding='SAME')

def conv_layer(inputx, size, i_chn, o_chn):
    W_conv = weight_variable([size, size, i_chn, o_chn])
    b_conv = bias_variable([o_chn])
    h_conv = tf.nn.relu(conv2d(inputx, W_conv) + b_conv)
    return h_conv

h_conv1 = conv_layer(x, 3, 1, 32)
h_pool1 = max_pool_2x2(h_conv1)
h_conv2 = conv_layer(h_pool1, 3, 32, 64)
h_pool2 = max_pool_2x2(h_conv2)
h_conv3 = conv_layer(h_pool2, 3, 64, 128)
h_pool3 = max_pool_2x2(h_conv3)
h_conv4 = conv_layer(h_pool3, 3, 128, 128)
h_pool4 = max_pool_1x2(h_conv4)

fc_dim = 512
pool_dim = 128

# W_fc1 = weight_variable([pool_dim, fc_dim])
# b_fc1 = bias_variable([fc_dim])
# h_pool3_flat = tf.reshape(h_pool3, [-1, pool_dim])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
h_pool3_flat = tf.nn.avg_pool(h_pool4, [1,h_pool4.get_shape()[1],h_pool4.get_shape()[2],1], [1,1,1,1],padding='VALID')
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_pool3_flat, keep_prob)
h_fc1 = tf.reshape(h_fc1_drop, [-1, pool_dim])

W_fc2 = weight_variable([pool_dim, 1])
b_fc2 = bias_variable([1])
y_conv= tf.matmul(h_fc1, W_fc2) + b_fc2

lambda_loss = 0.005
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
##loss
loss = tf.reduce_mean(tf.nn.l2_loss(y_conv-y_) + Lreg)
##rmse
error = tf.sqrt(tf.reduce_mean(tf.square(y_conv-y_)))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

# train -- (time, sensor)
trainmatrix=np.array(np.zeros((288*34,228)),dtype=float)
# test -- (record, time)
testmatrix=np.array(np.zeros((80*228,12)),dtype=float)

for i in range(34):
    train_df = pd.read_csv('../data/train/'+str(i)+'.csv', header=None).fillna(0)
    train_df.head()

    ### dataframe to array
    train = train_df.values
    for j in range(288):
        trainmatrix[i*288+j,:]=train[j,:]
        
for i in range(80):
    test_df = pd.read_csv('../data/test/'+str(i)+'.csv', header=None).fillna(0)
    test_df.head()

    ### dataframe to array
    test = test_df.values
    readmatrix=np.mat(np.zeros((12,228)),dtype=float)
    for j in range(12):
        readmatrix[j,:]=test[j,:]
    testmatrix[i*228:(i*228+228),:]=np.transpose(readmatrix)
    
dis_df = pd.read_csv('../data/distance.csv', header=None)
dis_df.head()
dis = dis_df.values

# trainmatrix.shape (9792, 228)
# testmatrix.shape  (18240, 12)

# train_max = np.max(trainmatrix)
# test_max = np.max(testmatrix)
# max_value = np.max([train_max, test_max])
# trainmatrix = trainmatrix/max_value
# testmatrix = testmatrix/max_value

mean = np.mean(trainmatrix)
std = np.std(trainmatrix)
trainmatrix = (trainmatrix-mean)/std
max_value = std
testmatrix = (testmatrix-mean)/std

#[sensor, neighbor] (228, neighbor+1)
near = np.array(np.zeros((228, k0+1)), dtype=int)
neardst = np.array(np.zeros((228, k0+1)))

for i in range(228):
    dist = list(dis[i])
    for j in range(k0+1):
        m = np.min(dist)
        k = dist.index(m)
        near[i,j] = k
        neardst[i,j] = m
        dist[k] = 10000000
    if near[i,0] != i:
        near[i,0], near[i,1] = near[i,1], near[i,0]
        
print(near) #(228, neighbor+1)

predict = [0 for i in range(testmatrix.shape[0]*3)] #15,30,45
Xgroup = [1 for i in range(trainmatrix.shape[0]-20)] #train, val
# 20: 训练标签，由v1预测v21，索引最大的训练样本是(全部-20)

# train & validate
for i in range(8000, trainmatrix.shape[0]-20):
    Xgroup[i] = 0
    
Xgroup = np.array(Xgroup)

err = 0
errx = 0
time_start = time.time()

time_len = trainmatrix.shape[0]-20
for i in range(1):
    
    ### 训练、验证、测试数据集 ###
    Xtrain = np.array(np.zeros((time_len, 12, k0+1, 1)), dtype=float)
    # Ytrain15 = np.array(np.zeros((time_len, 1)), dtype=float)
    # Ytrain30 = np.array(np.zeros((time_len, 1)), dtype=float)
    Ytrain45 = np.array(np.zeros((time_len, 1)), dtype=float)
    for j in range(time_len):
        Xtrain[j,:,0,0] = trainmatrix[j:(j+12), i]
        for l in range(k0):
            Xtrain[j,:,l+1,0]=trainmatrix[j:(j+12),near[i,l+1]]
        # Ytrain15[j,0] = trainmatrix[j+14,i]
        # Ytrain30[j] = trainmatrix[j+17,i]
        Ytrain45[j] = trainmatrix[j+20,i]
        
    # split
    X_train = Xtrain[Xgroup>=1]
    X_val = Xtrain[Xgroup==0]
    # Y_train15 = Ytrain15[Xgroup>=1]
    # Y_val15 = Ytrain15[Xgroup==0]
    # Y_train30 = Ytrain30[Xgroup>=1]
    # Y_val30 = Ytrain30[Xgroup==0]
    Y_train45 = Ytrain45[Xgroup>=1]
    Y_val45 = Ytrain45[Xgroup==0]
    
    # test
    Xtest = np.array(np.zeros((80,12,k0+1,1)),dtype=float)
    for j in range(80):
        Xtest[j,:,0,0] = testmatrix[j*228+i,:]
        for l in range(k0):
            Xtest[j,:,l+1,0]=testmatrix[j*228+near[i,l+1],:]

###### Initialize session ######
variables = tf.global_variables()
saver = tf.train.Saver(tf.global_variables())  
#sess = tf.Session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

out = 'out'
#out = 'out/%s_%s'%(model_name,'perturbation')
path1 = 'lr%r_batch%r_pre%r_epoch%r'%(lr,batch_size,pre_len,training_epoch)
path = os.path.join(out,path1)
if not os.path.exists(path):
    os.makedirs(path)
test_file = path1+'_record.txt'
batch_loss = []

totalbatch = int(X_train.shape[0]/batch_size)
for epoch in range(training_epoch):
    data_num = X_train.shape[0]
    index = np.arange(data_num) 
    np.random.shuffle(index)
    randomXtr = X_train[index]
    randomYtr = Y_train45[index]
    
    for m in range(totalbatch):
        mini_batch = randomXtr[m * batch_size : (m+1) * batch_size]
        mini_label = randomYtr[m * batch_size : (m+1) * batch_size]
        _, loss1, rmse1, train_output = sess.run([optimizer, loss, error, y_conv],
                                                 feed_dict = {x:mini_batch, y_:mini_label, keep_prob:0.6})

     # Test completely at every epoch
    loss2, rmse2, test_output = sess.run([loss, error, y_conv],
                                         feed_dict = {x:X_val, y_:Y_val45, keep_prob:1.0})
    print('Iter:{}'.format(epoch),
          'train_rmse:{:.4}'.format(rmse1 * max_value),
          'test_loss:{:.4}'.format(loss2),
          'test_rmse:{:.4}'.format(rmse2 * max_value))
    
    with open(test_file,'a+') as test_f:
        test_f.write("TESTER>> step: %d >> train_set rmse: %.4f >> valid_set rmse: %.4f\n"%(epoch, rmse1 * max_value, rmse2 * max_value))
    
    if (epoch % 100 == 0):
        #print("####### TEST OUTPUT #######", test_output)
        saver.save(sess, path+'/TGCN_pre_%r'%epoch, global_step = epoch)
        
time_end = time.time()
print(time_end-time_start,'s')