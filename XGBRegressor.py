
import xgboost as xgb
import sklearn
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
import numpy as np
from numpy import *
from numpy import mat
import time


trainmatrix=np.array(zeros((288*34,228)),dtype=float)
testmatrix=np.array(zeros((80*228,12)),dtype=float)

def as_num(x):
    l=len(x)
    y=[]
    for i in range(l):
        if str(x[i])[-4:]=="e+01" or str(x[i])[-5:-1]=="e+01":
            y.append(np.float32(x[i][:-5])*10)
        else:
            y.append(np.float32(x[i]))
    return(y)


for i in range(34):
    name="./data/train/"+str(i)+".csv"
    traincontext=open(name,"r")
    for j in range(288):
        speed=traincontext.readline()
        trainmatrix[i*288+j,:]=as_num((speed.split("\n")[0]).split(","))
        

for i in range(80):
    name="./data/test/"+str(i)+".csv"
    traincontext=open(name,"r")
    readmatrix=np.mat(zeros((12,228)),dtype=float)
    for j in range(12):
        speed=traincontext.readline()
        readmatrix[j,:]=as_num(speed.split(","))
    testmatrix[i*228:(i*228+228),:]=transpose(readmatrix)

distfile=open("./data/distance.csv","r")

def as_int(x):
    l=len(x)
    y=[]
    for i in range(l):
        if str(x[i])[-4:]=="e+03" or str(x[i])[-5:-1]=="e+03":
            y.append(int(float(x[i][:-5])*10000+0.5)/10)
            continue
        if str(x[i])[-4:]=="e+04" or str(x[i])[-5:-1]=="e+04":
            y.append(int(float(x[i][:-5])*100000+0.5)/10)
            continue
        if str(x[i])[-4:]=="e+05" or str(x[i])[-5:-1]=="e+05":
            y.append(int(float(x[i][:-5])*1000000+0.5)/10)
            continue
        if str(x[i])[-4:]=="e+02" or str(x[i])[-5:-1]=="e+02":
            y.append(int(float(x[i][:-5])*1000+0.5)/10)
            continue
        if str(x[i])[-4:]=="e+01" or str(x[i])[-5:-1]=="e+01":
            y.append(int(float(x[i][:-5])*10+0.5)/10)
            continue
        y.append(0)
    return(y)

k0=150  #15,25

near=np.array(zeros((228,k0+1)),dtype=int)
neardst=np.array(zeros((228,k0+1)),dtype=int)
for i in range(228):
    line=distfile.readline()
    dist=as_int(line.split(","))
    for j in range(k0+1):
        m=np.min(dist)
        k=dist.index(m)
        near[i,j]=k
        neardst[i,j]=m
        dist[k]=100000
    if near[i,0]!=i:
        near[i,0],near[i,1]=near[i,1],near[i,0]
    
print(near)

predict=[0 for i in range(54720)]
Xgroup=[1 for i in range(9112)]
for i in range(8000,9112):
    Xgroup[i]=0
Xgroup = np.array(Xgroup)

err=0
errx=0
time_start = time.time()

time_len = 9112 #268*34
for i in range(228):
    ### 训练、验证、测试数据集 ###
    Xtrain = np.array(np.zeros((time_len, 12*k0+12)), dtype=float)
    Ytrain15 = np.array(np.zeros((time_len)), dtype=float)
    Ytrain30 = np.array(np.zeros((time_len)), dtype=float)
    Ytrain45 = np.array(np.zeros((time_len)), dtype=float)
    Xtest=np.array(zeros((80,12*k0+12)),dtype=float)
    step=0
    for k in range(34):
        for j in range(k*288,(k+1)*288-20):
            Xtrain[step,0:12] = np.transpose(trainmatrix[j:(j+12), i])
            for l in range(k0):
                Xtrain[step, (l*12+12):(12*l+24)]=np.transpose(trainmatrix[j:(j+12),near[i,l+1]])
            Ytrain15[step] = trainmatrix[j+14,i]
            Ytrain30[step] = trainmatrix[j+17,i]
            Ytrain45[step] = trainmatrix[j+20,i]
            step+=1
    
    for j in range(80):
        for l in range(k0):
            Xtest[j,0:12]=testmatrix[j*228+i,:]
            Xtest[j,12*l+12:(12*l+24)]=testmatrix[j*228+near[i,(l+1)],:]

    other_params = {'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'objective': 'reg:squarederror', 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 0.1}

    Xval=Xtrain[Xgroup==0]
    Xtrain=Xtrain[Xgroup>=1]
    Yval15=Ytrain15[Xgroup==0]
    Ytrain15=Ytrain15[Xgroup>=1]
    Yval30=Ytrain30[Xgroup==0]
    Ytrain30=Ytrain30[Xgroup>=1]
    Yval45=Ytrain45[Xgroup==0]
    Ytrain45=Ytrain45[Xgroup>=1]
    
    model=xgb.XGBRegressor(**other_params)
    model.fit(Xtrain, Ytrain15)
    diff = model.predict(Xval) - Yval15
    errx += mean(diff*diff)
    lst1=model.predict(Xtest)
    for num in range(80):
        predict[3*228*num+i]=lst1[num]
    print(time.time()-time_start)
        
    model=xgb.XGBRegressor(**other_params)
    model.fit(Xtrain, Ytrain30)
    diff = model.predict(Xval) - Yval30
    errx += mean(diff*diff)
    lst2=model.predict(Xtest)
    for num in range(80):
        predict[3*228*num+228+i]=lst2[num]
    print(time.time()-time_start)
        
    model=xgb.XGBRegressor(**other_params)
    model.fit(Xtrain, Ytrain45)
    diff = model.predict(Xval) - Yval45
    errx += mean(diff*diff)
    lst3=model.predict(Xtest)
    for num in range(80):
        predict[3*228*num+228*2+i]=lst3[num]
    print(time.time()-time_start)
    
    print(" i = ", i)
    print(" errorx = ", (errx/3/(i+1))**0.5)
    
#     model = Lasso(alpha=1)
#     m1=model.fit(Xtrain, Ytrain15)
#     predicted1 = m1.predict(Xval)
#     diff=predicted1-transpose(Yval15).tolist()
#     #lst1=m1.predict(Xtest)
#     #for num in range(80):
#         #predict[3*228*num+i]=lst1[num]
#     err+=mean(diff*diff)
#     m2=model.fit(Xtrain, Ytrain30)
#     predicted2 = m2.predict(Xval)
#     diff=predicted2-transpose(Yval30).tolist()
#     #lst2=m2.predict(Xtest)
#     #for num in range(80):
#         #predict[3*228*num+228+i]=lst2[num]
#     err+=mean(diff*diff)
#     m3=model.fit(Xtrain, Ytrain45)
#     predicted3 = m3.predict(Xval)
#     diff=predicted3-transpose(Yval45).tolist()
# #     lst3=m3.predict(Xtest)
# #     for num in range(80):
# #         predict[3*228*num+228*2+i]=lst3[num]
#     err+=mean(diff*diff)
#     print(" error = ",(err/3/(i+1))**0.5)
    
import pandas as pd
dataframe = pd.DataFrame({'Expected':predict})
dataframe.to_csv("./print_predict%s.csv"%k0)

#     # XGB Regressor
#     other_params = {'learning_rate': 0.1,
#                     'n_estimators': 50,
#                     'max_depth': 5,
#                     'min_child_weight': 1, 
#                     'seed': 0,
#                     'objective': 'reg:squarederror', 
#                     'subsample': 0.8, 
#                     'colsample_bytree': 0.8,
#                     'gamma': 0.1, 
#                     'reg_alpha': 0.1, 
#                     'reg_lambda': 0.1}
#     cv_params = {'n_estimators': [25, 50, 75],'gamma': [0.1,0.2,0]}#,'min_child_weight': [0.5,1,1.5]}
#     ##### 15min
#     model = xgb.XGBRegressor(**other_params)
#     gs = sklearn.model_selection.GridSearchCV(estimator=model, param_grid=cv_params, n_jobs=1, scoring='neg_mean_squared_error', cv=3, verbose=1)
#     gs.fit(X_train, Y_train15)
#     print(gs.best_score_,gs.best_params_)#最好的得分,最好的参数
#     model = gs.best_estimator_ 
    
# #     model=xgb.XGBRegressor(**other_params)
# #     model.fit(X_train, Y_train15)
#     diff = model.predict(X_val) - Y_val15
#     print(X_val[0], Y_val15[0],model.predict(X_val)[0])
#     errx += np.mean(diff*diff)
#     print(errx)
#     lst1=model.predict(Xtest)
# #     lst1, errx = gs(X_train, Y_train15, X_val, Y_val15, Xtest, errx)
#     for num in range(80):
#         predict[3*228*num+i]=lst1[num]
#     print(time.time()-time_start)
    
#     ##### 30min
#     model = xgb.XGBRegressor(**other_params)
#     gs = sklearn.model_selection.GridSearchCV(estimator=model, param_grid=cv_params, n_jobs=1, scoring='neg_mean_squared_error', cv=3, verbose=1)
#     gs.fit(X_train, Y_train30)
#     print(gs.best_score_,gs.best_params_)#最好的得分,最好的参数
#     model = gs.best_estimator_ 
    
# #     model=xgb.XGBRegressor(**other_params)
# #     model.fit(X_train, Y_train30)
#     diff = model.predict(X_val) - Y_val30
#     errx += np.mean(diff*diff)
#     print(errx)
#     lst2=model.predict(Xtest)
#     for num in range(80):
#         predict[3*228*num+228+i]=lst2[num]
#     print(time.time()-time_start)
    
#     ##### 45min
#     model = xgb.XGBRegressor(**other_params)
#     gs = sklearn.model_selection.GridSearchCV(estimator=model, param_grid=cv_params, n_jobs=1, scoring='neg_mean_squared_error', cv=3, verbose=1)
#     gs.fit(X_train, Y_train45)
#     print(gs.best_score_,gs.best_params_)#最好的得分,最好的参数
#     model = gs.best_estimator_ 
    
# #     model=xgb.XGBRegressor(**other_params)
# #     model.fit(X_train, Y_train45)
#     diff = model.predict(X_val) - Y_val45
#     errx += np.mean(diff*diff)
#     print(errx)
#     lst3=model.predict(Xtest)
#     for num in range(80):
#         predict[3*228*num+228*2+i]=lst3[num]
#     print(time.time()-time_start)
    
#     print(" i = ", i)
#     print(" errorx = ", (errx/3/(i+1))**0.5)
    
# import pandas as pd
# dataframe = pd.DataFrame({'Expected':predict})
# dataframe.to_csv("./print_predict.csv")