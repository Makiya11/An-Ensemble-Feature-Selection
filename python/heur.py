import ast
import argparse
import sys
import pandas as pd
import time
import numpy as np
import os
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


def load_data(train_file, test_file):
    
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
                    
    X_train.drop(labels=correlated_features, axis=1, inplace=True)  
    X_test.drop(labels=correlated_features, axis=1, inplace=True)     
    
    return X_train, X_test

def importance():
    feat_labels = list(X_train)
    
    clf = RandomForestClassifier(n_estimators=100, random_state = 0)
    
    selection_start = time.time()
    clf = clf.fit(X_train, y_train)    
    selection_end = time.time()
    selection_time = selection_end - selection_start

    feature_importances = pd.DataFrame(clf.feature_importances_,
                                       index = X_train.columns,
                                        columns=['importance']).sort_values('importance',                
                                        ascending=False)
    data= []
    for i in range(1, X_train.shape[1]+1):
        data.append({'number of features':len(tuple(feat_labels)), 'features':list(tuple(feat_labels))})
        feat_labels.remove(feature_importances.index[-i])
    df_imp =  pd.DataFrame(data)
    
    if not os.path.exists('reduction'):
        os.makedirs('reduction')
    df_imp.to_csv('reduction/imp_reduction_'+train_file+'.csv',index=False)


def rfe():
    feat_labels = list(X_train)
    clf = RandomForestClassifier(n_estimators = 100, random_state=0)
    selection_start = time.time()
    data= []
    for i in range(X_train.shape[1],0,-1):
        rfe = RFE(clf, i)
        rfe.fit(X_train, y_train)
#         print(rfe.get_support(indices=True))
        
        features =list(map(lambda x:feat_labels[x], rfe.get_support(indices=True)))
        data.append({'number of features':len(features), 'features':list(tuple(features))})

#         print(features)
    df_rfe =  pd.DataFrame(data)
    if not os.path.exists('reduction'):
        os.makedirs('reduction')
    df_rfe.to_csv('reduction/rfe_reduction_'+train_file+'.csv',index=False)


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
    x['number of features'] = x.index 
    x['features'] = list(map(list, x['feature_names']))
    df_sbs = x[['number of features','features']]
    df_sbs = df_sbs.reset_index(drop=True)


    if not os.path.exists('reduction'):
        os.makedirs('reduction')
    df_sbs.to_csv('reduction/sbs_reduction_'+train_file+'.csv',index=False)

def univariate():
    feat_labels = list(X_train)
    
    data = []
    selection_start = time.time()
    for i in range(X_train.shape[1],0,-1):
        model = SelectKBest(chi2, k=i)
        model = model.fit(X_train, y_train)

        features =list(map(lambda x:feat_labels[x], model.get_support(indices=True)))
        data.append({'number of features':len(features), 'features':list(tuple(features))})
    
    selection_end = time.time()
    selection_time = selection_end - selection_start
    
    df_uni =  pd.DataFrame(data)
    if not os.path.exists('reduction'):
        os.makedirs('reduction')
    df_uni.to_csv('reduction/uni_reduction_'+train_file+'.csv',index=False)

def combine(df_uni, df_rfe, df_sbs, df_imp):
    
    df_all_fs = pd.DataFrame({'rfe':df_rfe['features'], 'sbs':df_sbs['features'], 'uni':df_uni['features'],'imp':df_imp['features']})
    return df_all_fs    


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

def cv(df_all_fs, classifer, clf, h):
    
    feat_label = X_train.columns
    # print(X_train.columns)
    feature_list = []
    data= []  
        
    for i in range(df_all_fs.shape[0]):
        features = {key: 0 for key in feat_label}
        for j in ast.literal_eval(df_all_fs['rfe'][i]):
            features[j] += 1
        for j in ast.literal_eval(df_all_fs['sbs'][i]):
            features[j] += 1
        for j in ast.literal_eval(df_all_fs['uni'][i]):
            features[j] += 1
        for j in ast.literal_eval(df_all_fs['imp'][i]):
            features[j] += 1
        
        f=[]
        for key, value in features.items():
            if value >=h:
                f.append(key)
        feature_list.append(list(tuple(f)))
        f.clear()

        
    for i in range(df_all_fs.shape[0]):
        try:
            global num_input_dim
            num_input_dim = len(feature_list[i])
            cv_start = time.time()
            cv_scores = cross_val_score(clf, X_train.loc[:,feature_list[i]] ,y_train,scoring='f1', cv=5)
            cv_end = time.time()
            cv_time = cv_end - cv_start
            cv_score_ = cv_scores
            max_score = cv_scores.mean()
            std = cv_scores.std()

        except:
            cv_score_=max_score=std= 0
            # print("An exception occurred")
        # print(cv_scores)
        
        data.append({'number of features':len(feature_list[i]), 'features':feature_list[i],'cv_scores':cv_score_, 'ave_score':max_score, 'std_dev':std})

    df =  pd.DataFrame(data)
    if not os.path.exists('cv'):
        os.makedirs('cv')
    df.to_csv('cv/'+str(classifer)+'_heur_cv_'+str(h)+train_file+'.csv',index=False)

def do_ml(df_all_fs, classifer, clf, h):
    
    
    feat_label = X_train.columns
    print(X_train.columns)
    feature_list = []
    data= []  
    print(df_all_fs)
    for i in range(df_all_fs.shape[0]):
        features = {key: 0 for key in feat_label}
        for j in ast.literal_eval(df_all_fs['rfe'][i]):
            features[j] += 1
        for j in ast.literal_eval(df_all_fs['sbs'][i]):
            features[j] += 1
        for j in ast.literal_eval(df_all_fs['uni'][i]):
            features[j] += 1
        for j in ast.literal_eval(df_all_fs['imp'][i]):
            features[j] += 1
        
        f=[]
        for key, value in features.items():
            if value >=h:
                f.append(key)
        feature_list.append(list(tuple(f)))
        f.clear()

        
    for i in range(df_all_fs.shape[0]):
        try:
            global num_input_dim
            num_input_dim = len(feature_list[i])
            train_start = time.time()
            clf.fit(X_train.loc[:,feature_list[i]], y_train)
            train_end = time.time()
            train_time = train_end - train_start

            test_start = time.time()
            y_pred = clf.predict(X_test.loc[:,feature_list[i]])
            test_end = time.time()
            test_time = test_end - test_start

            accuracy = accuracy_score(y_test, y_pred)

            # Making the Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            tn = cm[0][0]
            tp = cm[1][1]
            fp = cm[0][1]
            fn = cm[1][0]

            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            f1_score = 2*((precision*recall)/(precision+recall))

        except:
            f1_score_list=train_time=test_time=accuracy=tn=tp=fn=fp=precision=recall=f1_score = 0
            # print("An exception occurred")
            
        data.append({'number of features':len(feature_list[i]), 'features':feature_list[i], 'train_time':train_time, 'test_time':test_time, 'accuracy':accuracy,
                 'tn':tn,'tp':tp,'fn':fn,'fp':fp,'precision':precision,'recall':recall,'F-1score':f1_score})
    order = ['number of features','features','train_time','test_time','accuracy',
                 'tn','tp','fn','fp','precision','recall','F-1score']
    df =  pd.DataFrame(data)
    df = df[order]
    
    if not os.path.exists('test'):
        os.makedirs('test')
    df.to_csv('test/'+str(classifer)+'_heur_test_'+str(h)+train_file+'.csv',index=False)



def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-tr", "--train", help="Your input train file.")
    parser.add_argument("-te", "--test", help="Your input test file.")
    parser.add_argument("-m", "--classifer", help="Machine learning classier.")
    options = parser.parse_args(args)
    return options

options = getOptions(sys.argv[1:])

print(options.train)
print(options.classifer)

train_file = options.train
test_file = options.test
classifer = options.classifer

X_train, X_test, y_train, y_test = load_data()
X_train, X_test = feature_scaling(X_train, X_test)
X_train, X_test = filter_method(X_train,X_test)
if not os.path.exists('reduction/imp_reduction_'+train_file+'.csv'):
    importance()
df_imp = pd.read_csv('reduction/imp_reduction_'+train_file+'.csv')

    
if not os.path.exists('reduction/uni_reduction_'+train_file+'.csv'):
    univariate()
df_uni = pd.read_csv('reduction/uni_reduction_'+train_file+'.csv')

    
if not os.path.exists('reduction/rfe_reduction_'+train_file+'.csv'):
    rfe()
df_rfe = pd.read_csv('reduction/rfe_reduction_'+train_file+'.csv')

    
if not os.path.exists('reduction/sbs_reduction_'+train_file+'.csv'):
    sbs()
df_sbs = pd.read_csv('reduction/sbs_reduction_'+train_file+'.csv')

df_all_fs = combine(df_uni, df_rfe, df_sbs, df_imp)

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
    do_ml(df_all_fs, classifer, clf, 1)
    cv(df_all_fs, classifer, clf, 1)
    
    do_ml(df_all_fs, classifer, clf, 3)
    cv(df_all_fs, classifer, clf, 3)
    
    do_ml(df_all_fs, classifer, clf, 4)
    cv(df_all_fs, classifer, clf, 4)