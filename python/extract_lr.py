#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:38:51 2019

@author: makiya
"""

import pandas as pd
import ast


def allfeatures_rsui():
    rfe = pd.read_csv("../cv/cv_rfe_01.csv")
    
    sbs = pd.read_csv("../cv/cv_sbs_01.csv")
    
    uni = pd.read_csv("../cv/cv_uni_01.csv")
    
    imp = pd.read_csv("../cv/cv_imp_01.csv")
    
    df_allrsui = pd.DataFrame({'rfe':rfe['features'], 'sbs':sbs['features'], 'uni':uni['features'],'imp':imp['features']})
    return df_allrsui     



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    
    for i in range(len(correlation_matrix.columns)):  
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
                
    print(correlated_features)  
    
    
    # Drop Correlated data
    X_train.drop(labels=correlated_features, axis=1, inplace=True)  
    X_test.drop(labels=correlated_features, axis=1, inplace=True)
    num_feature = X_train.shape[1]  
     
    
    return X_train, X_test



def do_ml(df_allrsui, X_train, X_test, h):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(solver = 'liblinear',random_state=0)
    
    
    feat_label = X_train.columns
    n=df_allrsui.shape[0]
    f1_score_list =[]
    num_feature=[]
    feature_list = []
    train_times = []
    test_times = []
    
        
    for i in range(n):
        features = {key: 0 for key in feat_label}
        for j in ast.literal_eval(df_allrsui['rfe'][i]):
            features[j] += 1
        for j in ast.literal_eval(df_allrsui['sbs'][i]):
            features[j] += 1
        for j in ast.literal_eval(df_allrsui['uni'][i]):
            features[j] += 1
        for j in ast.literal_eval(df_allrsui['imp'][i]):
            features[j] += 1
        
        f=[]
        for key, value in features.items():
            if value >=h:
                f.append(key)
        print(f)
        feature_list.append(tuple(f))
        num_feature.append(len(f))
        f.clear()
    
    for i in range(len(feature_list)):
        feature_list[i]  = list(feature_list[i]) 
        
    for i in range(n):
        try:
            print(feature_list[i])
            train_start = time.time()
            classifier.fit(X_train.loc[:,feature_list[i]], y_train)
            train_end = time.time()
            train_time = train_end - train_start
            train_times.append(train_time)
            
            test_start = time.time()
            y_pred = classifier.predict(X_test.loc[:,feature_list[i]])
            test_end = time.time()
            test_time = test_end - test_start
            test_times.append(test_time)
            
            from sklearn.metrics import accuracy_score 
            accuracy = accuracy_score(y_test, y_pred)
            
            # Making the Confusion Matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            tn = cm[0][0]
            tp = cm[1][1]
            fn = cm[1][0]
            fp = cm[0][1]

            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            f1_score = 2*((precision*recall)/(precision+recall))
            print(f1_score)
            f1_score_list.append(f1_score)

        except:
            f1_score_list.append(0)
            train_times.append(0)
            test_times.append(0)
            print("An exception occurred")
    return f1_score_list, num_feature, feature_list, train_times, test_times


df_allrsui = allfeatures_rsui()
X_train, X_test, y_train, y_test= load_data()
X_train, X_test = feature_scaling(X_train, X_test, y_train, y_test)
X_train, X_test = filter_method(X_train,X_test)
f1_score_one, num_one, one_features, train_time_one, test_time_one = do_ml(df_allrsui, X_train, X_test, 1)
#f1_score_two, num_two, two_features, train_time_two, test_time_two = feature_rand_forest(df_allrsui, X_train, X_test, 2)
f1_score_three, num_three, three_features, train_time_three, test_time_three = do_ml(df_allrsui, X_train, X_test, 3)
f1_score_four, num_four, four_features, train_time_four, test_time_four = do_ml(df_allrsui, X_train, X_test, 4)

#'num_two':num_two, 'two feature':two_features,'f1_score two':f1_score_two,
df_iandu = pd.DataFrame({'num_one':num_one, 'one feature':one_features,'f1_score one':f1_score_one,'train-time-one':train_time_one,'test-time-one':test_time_one,
                         'num_three':num_three, 'three feature':three_features, 'f1_score three': f1_score_three,'train-time-three':train_time_three,'test-time-three':test_time_three,
                         'num_four':num_four, 'four feature':four_features, 'f1_score four': f1_score_four,'train-time-four':train_time_four,'test-time-four':test_time_four})
df_iandu.to_csv('../common/ids_lr_01.csv',index=False)
