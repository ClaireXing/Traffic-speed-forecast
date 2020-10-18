# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:15:50 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
import pickle as pkl

def load_sz_data(dataset):
    sz_adj = pd.read_csv(r'data/sz_adj.csv',header=None)
    adj = np.mat(sz_adj)
    sz_tf = pd.read_csv(r'data/sz_speed.csv')
    return sz_tf, adj

def load_los_data(dataset):
    los_adj = pd.read_csv(r'data/los_adj.csv',header=None)
    adj = np.mat(los_adj)
    los_tf = pd.read_csv(r'data/los_speed.csv')
    return los_tf, adj

def load_cal_data(dataset):
    # train -- (time, sensor)
    trainmatrix=np.array(np.zeros((288*34,228)),dtype=float)
    for i in range(34):
        train_df = pd.read_csv('../data/train/'+str(i)+'.csv', header=None)
        
        ### dataframe to array
        train = np.mat(train_df)# train_df.values
        for j in range(288):
            trainmatrix[i*288+j,:]=train[j,:]
    
    dis_df = pd.read_csv('../data/distance.csv', header=None)
    dis = np.mat(dis_df)
    
    return trainmatrix, dis

def load_test_data(dataset):
    testmatrix=np.array(np.zeros((80,12,228)),dtype=float)
    for i in range(80):
        test_df = pd.read_csv('../data/test/'+str(i)+'.csv', header=None).fillna(0)

        ### dataframe to array
        readmatrix=np.mat(test_df)
        testmatrix[i,:,:]=readmatrix
    return testmatrix

def preprocess_data(data, time_len, rate, seq_len, pre_len):
    X, Y = [], []
    for k in range(34):
        for i in range(k*288,(k+1)*288-seq_len-pre_len+1):
            a = data[i:i+seq_len+pre_len]
            X.append(a[0:seq_len])
            Y.append(a[-1])
    
    time_len = len(X)
    train_size = int(time_len * rate)
    trainX = X[0:train_size]
    trainY = Y[0:train_size]
    testX = X[train_size:time_len]
    testY = Y[train_size:time_len]
#     train_data = data[0:train_size]
#     test_data = data[train_size:time_len]
    
#     trainX, trainY, testX, testY = [], [], [], []
#     for k in range(34):
#         for i in range(len(train_data) - seq_len - pre_len):
#             a = train_data[i: i + seq_len + pre_len]
#             trainX.append(a[0 : seq_len])
#     #         #trainY.append(a[seq_len : seq_len + pre_len])
#             trainY.append(a[-1])
#     #         y15 = a[seq_len + 2]
#     #         y30 = a[seq_len + 5]
#     #         y45 = a[seq_len + 8]
#     #         y = np.concatenate((y15,y30),axis=0)
#     #         y = np.concatenate((y,y45),axis=0)
#     #         trainY.append(y)
#         for i in range(len(test_data) - seq_len -pre_len):
#             b = test_data[i: i + seq_len + pre_len]
#             testX.append(b[0 : seq_len])
#             #testY.append(b[seq_len : seq_len + pre_len])
#             testY.append(b[-1])
# #         y15 = b[seq_len + 2]
# #         y30 = b[seq_len + 5]
# #         y45 = b[seq_len + 8]
# #         y = np.concatenate((y15,y30),axis=0)
# #         y = np.concatenate((y,y45),axis=0)
# #         testY.append(y)
      
    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    return trainX1, trainY1, testX1, testY1