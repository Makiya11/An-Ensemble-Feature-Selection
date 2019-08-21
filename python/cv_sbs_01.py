#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 23:38:54 2019

@author: makiya
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import time

# Importing the training dataset
def load_data():
    
    train_dataset = pd.read_csv("../../IDS2017/IDS2017_01.csv")
    test_dataset = pd.read_csv("../../IDS2017/IDS2017_02.csv")

    # create Xtrain and ytrain
    X_train = pd.DataFrame(train_dataset.iloc[:, : -1].values, columns = train_dataset.columns[:-1])
    y_train = train_dataset.iloc[:, -1].values
    
    # create Xtest and ytest
    X_test = pd.DataFrame(test_dataset.iloc[:, : -1].values, columns = test_dataset.columns[:-1])
    y_test = test_dataset.iloc[:, -1].values
    return X_train, X_test, y_train, y_test  
    

def num_fix(ticker):
     if ticker > 1:
         return 1
     else:
         return ticker

# feature scalling
def feature_scaling(X_train, X_test, y_train, y_test):
    from sklearn import preprocessing
    minmaxscaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    X_train = pd.DataFrame(minmaxscaler.fit_transform(X_train), columns = X_train.columns[:])
    X_test = pd.DataFrame(minmaxscaler.transform(X_test), columns = X_test.columns[:])
    for i in  X_test.columns[:]:
        X_test[i] = X_test[i].apply(num_fix)
    return X_train, X_test

def filter_method(X_train,X_test):  
    
    #Removing Correlated Features
    correlated_features = set()  
    correlation_matrix = X_train.corr() 
    
    for i in range(len(correlation_matrix .columns)):  
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
                
    print(correlated_features)  
    
    
    # Drop Correlated data
    X_train.drop(labels=correlated_features, axis=1, inplace=True)  
    X_test.drop(labels=correlated_features, axis=1, inplace=True)
     
    
    return X_train, X_test

def sbs():
    # Build RF classifier to use in feature selection
    clf = RandomForestClassifier(n_estimators=100, random_state = 0)
    
    # Build step backward feature selection
    sbs1 = sfs(clf,
               k_features=1,
               forward=False, 
               floating=False,
               scoring='accuracy',
               cv=0)
    
    selection_start = time.time()
    # Perform SbS
    sbs1 = sbs1.fit(X_train, y_train)
    selection_end = time.time()
    selection_time = selection_end - selection_start

    x=pd.DataFrame.from_dict(sbs1.get_metric_dict()).T
    n = [num+1 for num in range(X_train.shape[1])]
    n.reverse()
    
    for i in range(x.shape[0]):
        x['feature_names'][i+1] = list(x['feature_names'][i+1])
    
    df = pd.DataFrame({'number of features':n, 'features':x['feature_names'], 'selection time':selection_time })
    
    df.to_csv('../reduction/sbs_01.csv',index=False)
    return sbs1, df

def combination(df):
    combs= []
    for i in range(X_train.shape[1]):
        combs.append(list(df.iloc[i,1]))
    return combs

def cv_rf(combs):
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    n = len(combs)
    scores=[]
    average=[]
    standard_deviation = []
    times =[]

    for i in range (len(combs)):
       rf = RandomForestClassifier(n_estimators = 100, random_state=0)

       cv_start = time.time()
       cv_scores = cross_val_score(rf,X_train.loc[:,combs[i]] ,y_train, cv=5)
       cv_end = time.time()
       cv_time = cv_end - cv_start
       print(scores)
       print(cv_time)
       
       scores.append(cv_scores)
       average.append(cv_scores.mean())
       standard_deviation.append(cv_scores.std())
       times.append(cv_time)
       
       print(combs[i])
    n = [[num+1] for num in range(n)]
    n.reverse()

    
    df = pd.DataFrame({'number of features':n, 'features':combs,'cv_scores':scores, 'ave_score':average, 'std_dev':standard_deviation, 'cv_time':times})

     
    df.to_csv('../cv/cv_sbs_01.csv',index=False)

def rf(combs):
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.ensemble import RandomForestClassifier

    inf=[[] for i in range(len(combs))]
    for i in range (len(combs)): 
        rf = RandomForestClassifier(n_estimators = 100, random_state=0)

        train_start = time.time()
        rf.fit(X_train.loc[:,combs[i]], y_train)
        train_end = time.time()
        train_time = train_end - train_start
        print(train_time)

        test_start = time.time()
        y_pred = rf.predict(X_test.loc[:,combs[i]])
        test_end = time.time()
        test_time = test_end - test_start
        print(test_time)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)
        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn = cm[0][0]
        tp = cm[1][1]
        fn = cm[1][0]
        fp = cm[0][1]
       
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1_score = 2*((precision*recall)/(precision+recall))
        print('precision = ', precision)
        print('recall = ', recall)
        print('F1-score = ', f1_score)
        print(combs[i])
        inf[i].extend([train_time,test_time,accuracy,tn,tp,fp,fn,precision,recall,f1_score])
    numf = [[i+1] for i in range(len(combs))]
    numf.reverse()
    df = pd.DataFrame(columns=['num features','features','train_time','test_time','accuracy','tn','tp','fp','fn','precision','recall','F-1score'])
    for i in range (len(combs)): 
        df.loc[i] = numf[i] + [combs[i]] + inf[i]
    df.to_csv('../test/rf_sbs_01.csv',index=False)
    return inf



# # variable selection part
X_train, X_test, y_train, y_test = load_data()
X_train, X_test = feature_scaling(X_train, X_test, y_train, y_test)
X_train, X_test = filter_method(X_train,X_test)
sbs1, df = sbs()


# cross-validation part
X_train, X_test, y_train, y_test = load_data()
X_train, X_test = feature_scaling(X_train, X_test, y_train, y_test)
X_train, X_test = filter_method(X_train,X_test)
combs = combination(df)
cv_rf(combs)
  
# testing part
rf(combs)
