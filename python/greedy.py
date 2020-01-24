import os
import argparse
import sys
import pandas as pd
import time
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from keras import backend
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression



def load_data():
    
    train_dataset = pd.read_csv(train_file, nrows=100)
    test_dataset = pd.read_csv(test_file, nrows=100)

    train_dataset = train_dataset._get_numeric_data()
    test_dataset = test_dataset._get_numeric_data()
    
    X_train = pd.DataFrame(train_dataset.iloc[:, : -1].values, columns = train_dataset.columns[:-1])
    y_train = train_dataset.iloc[:, -1].values
    
    X_test = pd.DataFrame(test_dataset.iloc[:, : -1].values, columns = test_dataset.columns[:-1])
    y_test = test_dataset.iloc[:, -1].values
    return X_train, X_test, y_train, y_test 
    

def num_fix(ticker):
    if ticker > 1:
         return 1
    else:
         return ticker

def feature_scaling(X_train, X_test):
   
    minmaxscaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    X_train = pd.DataFrame(minmaxscaler.fit_transform(X_train), columns = X_train.columns[:])
    X_test = pd.DataFrame(minmaxscaler.transform(X_test), columns = X_test.columns[:])
    for i in  X_test.columns[:]:
        X_test[i] = X_test[i].apply(num_fix)
    return X_train, X_test

def filter_method(X_train,X_test):  
    
    correlated_features = set()  
    correlation_matrix = X_train.corr() 
    
    for i in range(len(correlation_matrix .columns)):  
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
                
    print(correlated_features)  
    
    X_train.drop(labels=correlated_features, axis=1, inplace=True)  
    X_test.drop(labels=correlated_features, axis=1, inplace=True)     
    
    return X_train, X_test

def importance(num_features, features):
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    selection_start = time.time()
    clf = clf.fit(X_train.loc[:,features], y_train)    
    selection_end = time.time()
    selection_time = selection_end - selection_start
    
    feature_list = list(features)
    
    feature_importances = pd.DataFrame(clf.feature_importances_,
                                   index = features,
                                    columns=['importance']).sort_values('importance',                
                                    ascending=False)
    
    feature_list.remove(feature_importances.index[-1])

    return feature_list


def rfe(num_features, features):
    clf= RandomForestClassifier(n_estimators = 100, random_state=0)
    feat_labels = list(X_train.loc[:,features])

    selection_start = time.time()
    rfe = RFE(clf, num_features)
    rfe.fit(X_train.loc[:,features], y_train)
        
    feature_list=[]
    for feature_list_index in rfe.get_support(indices=True):
            feature_list.append(feat_labels[feature_list_index])
#            print(feat_labels[feature_list_index])
    selection_end = time.time()
    selection_time = selection_end - selection_start
    
    return feature_list

def univariate(num_features, features):


    feat_labels = list(X_train.loc[:,features])
    selection_start = time.time()
    model = SelectKBest(chi2, k=num_features)
    model =model.fit(X_train.loc[:,features], y_train)

    feature_list=[]
    for feature_list_index in model.get_support(indices=True):
            feature_list.append(feat_labels[feature_list_index])
#            print(feat_labels[feature_list_index])
    selection_end = time.time()
    selection_time = selection_end - selection_start    
 
    return feature_list



def sbs(num_features, features):
    # Build RF classifier to use in feature selection
    clf = RandomForestClassifier(n_estimators=100, random_state = 0)
        # Build step backward feature selection
    sbs1 = sfs(clf,
               k_features=num_features,
               forward=False, 
               floating=False,
               scoring='f1',
               cv=0)
    
    selection_start = time.time()
    # Perform SbS
    sbs1 = sbs1.fit(X_train.loc[:,features], y_train)
    selection_end = time.time()
    selection_time = selection_end - selection_start
    feature_list = list(sbs1.subsets_[num_features]['feature_names'])
    return feature_list



def create_network():
    model = Sequential()
    model.add(Dense(50, kernel_initializer ='uniform', activation = 'relu', input_dim = num_input_dim))
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer ='uniform', activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer ='uniform', activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer ='uniform', activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer ='uniform',activation = 'sigmoid'))
    
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def greedy_approach(clf, classifer):
    data = []
    features = list(X_train.columns)
    
    for num in range(X_train.shape[1], 0, -1):
        fs_imp = importance(num, features)
        fs_rfe = rfe(num, features)
        fs_uni = univariate(num, features)
        fs_sbs = sbs(num, features)
        
        fs_list = [fs_imp,fs_rfe,fs_uni,fs_sbs]
        
        for i in range(len(fs_list)):
            for j in range(i+1,len(fs_list)):
                if fs_list[i] == fs_list[j]:
#                     print(str(fs_list[i])+'is same as'+str(fs_list[j]))
#                     print('----------------------------------------------------------')
                    fs_list[i] = 0
          
        max_score = 0
        for fs in fs_list:
            try:
                # print(fs)
                global num_input_dim
                num_input_dim = len(fs)
                cv_start = time.time()
                cv_scores = cross_val_score(clf,X_train.loc[:,fs], y_train, scoring='f1', cv=5)
                cv_end = time.time()
                cv_time = cv_end - cv_start
                # print(cv_time)
                backend.clear_session()
                print(cv_scores)
            except:
#                 print('exception occour')
                backend.clear_session()
                continue
            print(str(max_score)+' ? '+str(cv_scores.mean()))
            if max_score < cv_scores.mean():
                # print(str(max_score)+' < '+str(cv_scores.mean()))
                cv_score_ = cv_scores
                max_score = cv_scores.mean()
                std = cv_scores.std()
                features = fs
        data.append({'number of features':num, 'features':features,'cv_scores':cv_score_, 'ave_score':max_score, 'std_dev':std})
        
    order = ['number of features','features','cv_scores', 'ave_score', 'std_dev']
    df =  pd.DataFrame(data)
    df = df[order]
    if not os.path.exists('cv'):
        os.makedirs('cv')
    df.to_csv('cv/'+str(classifer)+'_greedy_cv_'+train_file+'.csv',index=False)
    return df
    
def do_ml(df, clf, classifer):
    
    data= []   
    for i in range(df.shape[0]):
        global num_input_dim
        num_input_dim = len(df['features'][i])
        # print(df['features'][i])
        train_start = time.time()
        clf.fit(X_train.loc[:,df['features'][i]], y_train)
        train_end = time.time()
        train_time = train_end - train_start
        
        test_start = time.time()
        y_pred = clf.predict(X_test.loc[:,df['features'][i]])
        test_end = time.time()
        test_time = test_end - test_start
        
        accuracy = accuracy_score(y_test, y_pred)
        
        cm = confusion_matrix(y_test, y_pred)
        tn = cm[0][0]
        tp = cm[1][1]
        fp = cm[0][1]
        fn = cm[1][0]
        
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1_score = 2*((precision*recall)/(precision+recall))
        backend.clear_session()
        data.append({'number of features':df['number of features'][i], 'features':df['features'][i], 'train_time':train_time, 'test_time':test_time, 'accuracy':accuracy,
                 'tn':tn,'tp':tp,'fn':fn,'fp':fp,'precision':precision,'recall':recall,'F-1score':f1_score})
    order = ['number of features','features','train_time','test_time','accuracy',
                 'tn','tp','fn','fp','precision','recall','F-1score']
    df =  pd.DataFrame(data)
    df = df[order]
    if not os.path.exists('test'):
        os.makedirs('test')
    df.to_csv('test/'+str(classifer)+'_greedy'+train_file+'.csv',index=False)

def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-tr", "--train", help="Your input train file.")
    parser.add_argument("-te", "--test", help="Your input test file.")
    parser.add_argument("-m", "--classifer", help="Machine learning classier.")
    options = parser.parse_args(args)
    return options  

options = getOptions(sys.argv[1:])

train_file = options.train
test_file = options.test
classifer = options.classifer

X_train, X_test, y_train, y_test = load_data()
X_train, X_test = feature_scaling(X_train, X_test)
X_train, X_test = filter_method(X_train,X_test)

if classifer.upper() =='DNN':
    clf = KerasClassifier(build_fn=create_network, epochs=15, batch_size = 100)
elif classifer.upper() =='RF':
    clf = RandomForestClassifier(n_estimators=100, random_state = 0)
elif classifer.upper() =='GB':
    clf = GradientBoostingClassifier(n_estimators=200, random_state = 0)
elif classifer.upper() =='LR':
    clf = LogisticRegression(solver = 'liblinear',C=100,random_state=0)
else:
    print('Please input DNN or RF or GB or LR')


if classifer=='DNN' or classifer == 'RF' or classifer == 'GB' or classifer == 'LR':
    df = greedy_approach(clf, classifer)
    do_ml(df, clf, classifer)

