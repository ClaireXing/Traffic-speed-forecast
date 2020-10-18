import numpy as np
from numpy import mat
import matplotlib.pyplot as plt  
import sklearn
from sklearn.neural_network import MLPRegressor
import time
import pandas as pd

# train -- (time, sensor)
trainmatrix=np.array(np.zeros((288*34,228)),dtype=float)
# test -- (record, time)
testmatrix=np.array(np.zeros((80*228,12)),dtype=float)

for i in range(34):
    train_df = pd.read_csv('./data/train/'+str(i)+'.csv', header=None).fillna(0)
    train_df.head()

    ### dataframe to array
    train = train_df.values
    for j in range(288):
        trainmatrix[i*288+j,:]=train[j,:]
        
for i in range(80):
    test_df = pd.read_csv('./data/test/'+str(i)+'.csv', header=None).fillna(0)
    test_df.head()

    ### dataframe to array
    test = test_df.values
    readmatrix=np.mat(np.zeros((12,228)),dtype=float)
    for j in range(12):
        readmatrix[j,:]=test[j,:]
    testmatrix[i*228:(i*228+228),:]=np.transpose(readmatrix)
    
dis_df = pd.read_csv('./data/distance.csv', header=None)
dis_df.head()
dis = dis_df.values

print(trainmatrix.shape)
print(testmatrix.shape)
print(dis.shape)

train_max = np.max(trainmatrix)
test_max = np.max(testmatrix)
max_value = np.max([train_max, test_max])
trainmatrix = trainmatrix/max_value
testmatrix = testmatrix/max_value

k0 = 20 # num of neighbors

#[sensor, neighbor]
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
        
print(near)

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
for i in range(228):
    
    ### 训练、验证、测试数据集 ###
    Xtrain = np.array(np.zeros((time_len, 12*k0+12)), dtype=float)
    Ytrain15 = np.array(np.zeros((time_len)), dtype=float)
    Ytrain30 = np.array(np.zeros((time_len)), dtype=float)
    Ytrain45 = np.array(np.zeros((time_len)), dtype=float)
    for j in range(time_len):
        Xtrain[j,0:12] = np.transpose(trainmatrix[j:(j+12), i])
        for l in range(k0):
            Xtrain[j, (l*12+12):(12*l+24)]=np.transpose(trainmatrix[j:(j+12),near[i,l+1]])
        Ytrain15[j] = trainmatrix[j+14,i]
        Ytrain30[j] = trainmatrix[j+17,i]
        Ytrain45[j] = trainmatrix[j+20,i]
        
    # split
    X_train = Xtrain[Xgroup>=1]
    X_val = Xtrain[Xgroup==0]
    Y_train15 = Ytrain15[Xgroup>=1]
    Y_val15 = Ytrain15[Xgroup==0]
    Y_train30 = Ytrain30[Xgroup>=1]
    Y_val30 = Ytrain30[Xgroup==0]
    Y_train45 = Ytrain45[Xgroup>=1]
    Y_val45 = Ytrain45[Xgroup==0]
    
    # test
    Xtest = np.array(np.zeros((80,12*k0+12)),dtype=float)
    for j in range(80):
        Xtest[j,0:12] = testmatrix[j*228+i,:]
        for l in range(k0):
            Xtest[j,(12*l+12):(12*l+24)]=testmatrix[j*228+near[i,l+1],:]
    
    # XGB Regressor
    other_params = {'learning_rate': 0.1,
                    'n_estimators': 50,
                    'max_depth': 5,
                    'min_child_weight': 1, 
                    'seed': 0,
                    'objective': 'reg:squarederror', 
                    'subsample': 0.8, 
                    'colsample_bytree': 0.8,
                    'gamma': 0, 
                    'reg_alpha': 0.1, 
                    'reg_lambda': 0.1}
    
#     cv_params = {'n_estimators': [400, 500, 600, 700]}
#     model = xgb.XGBRegressor(**other_params)
#     gs = sklearn.model_selection.GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5, verbose=1)
#     gs.fit(X_train, Y_train15)
    
#     print(gs.best_score_)#最好的得分
#     print(gs.best_params_)#最好的参数
    #model=xgb.XGBRegressor(**other_params)
    #model.fit(X_train, Y_train15)
    model = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', solver='adam', alpha=1e-04, 
              learning_rate_init=0.001, max_iter=1000)
    model.fit(X_train, Y_train15)
    diff = (model.predict(X_val) - Y_val15)*max_value
    errx += np.mean(diff*diff)
    lst1=model.predict(Xtest)*max_value
    for num in range(80):
        predict[3*228*num+i]=lst1[num]
    print(time.time()-time_start)

    #model=xgb.XGBRegressor(**other_params)
    model = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', solver='adam', alpha=1e-04, 
              learning_rate_init=0.001, max_iter=1000)
    model.fit(X_train, Y_train30)
    diff = (model.predict(X_val) - Y_val30)*max_value
    errx += np.mean(diff*diff)
    lst2=model.predict(Xtest)*max_value
    for num in range(80):
        predict[3*228*num+228+i]=lst2[num]
    print(time.time()-time_start)

    #model=xgb.XGBRegressor(**other_params)
    model = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', solver='adam', alpha=1e-04, 
              learning_rate_init=0.001, max_iter=1000)
    model.fit(X_train, Y_train45)
    diff = (model.predict(X_val) - Y_val45)*max_value
    errx += np.mean(diff*diff)
    lst3=model.predict(Xtest)*max_value
    for num in range(80):
        predict[3*228*num+228*2+i]=lst3[num]
    print(time.time()-time_start)

    print(" i = ", i)
    print(" errorx = ", (errx/3/(i+1))**0.5)
    
import pandas as pd
dataframe = pd.DataFrame({'Expected':predict})
dataframe.to_csv("./print_predict_mlp.csv")

## XGB 6.70(#20 #10) 6.83 #5
## XGB 1/d 6.60