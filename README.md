# AN ENSEMBLE FEATURE SELECTION

The deatail is in the PDF file above. 


# Instruction Manual
 
Technology requirements

Python 3.6
Keras 2.2.4  
Tensorflow 1.14.0 
Sklearn 0.0
Mlxtend 0.16.0
Pandas 0.24.2
Numpy 1.16.4
 
Download data:

UNSW NB15 https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/
IDS2017 https://www.unb.ca/cic/datasets/ids-2017.html
 

## Ensemble Feature Selection:

## Heuristic and Greedy methods
**Note:**
Categorical values(non-numerical values) and unnecessary(ex, ID) data have to be removed before using this method.
In training and testing files, the last column has to be the label. If the last column is not the label, you need to fix it before implementing this program.

**Command**
python heur.py( or greedy.py) -tr TRAINING FILENAME -teTESTING FILENAME -m CLASSIFIER
 
Command Example
python heur.py -tr IDS2017_01.csv -te IDS2017_02.csv -m LR
python heur.py -tr IDS2017_03.csv -te IDS2017_04.csv -m DNN
python greedy.py -tr UNSW_NB15_training-set.csv -te UNSW_NB15_testing-set.csv -m GB

heur.py will create reduction, cv, test folder and files.
reduction folder contains each feature selection’s selected feature results
cv folder contains the cross-validation results
test folder contains the actual test results
Filename: CLASSIFIERNAME_METHOD_FILENAME.csv (heur_1 means Union, heur_3 means Quorum, heur_4 means Intersection)

greedy.py will create reduction, cv, test folder, and files.
cv folder contains the cross-validation results
test folder contains the actual test results

% python heur.py -h 
usage: heur.py [-h] [-tr TRAIN] [-te TEST] [-m CLASSIFIER]
optional arguments:
  -h, --help        	show this help message and exit
  -tr TRAIN, --train TRAIN
                    	Your input train file.
  -te TEST, --test TEST
                    	Your input test file.
  -m CLASSIFIER, --classifer CLASSIFIER
                    	Machine learning classier.

% python greedy.py -h 
usage: greedy.py [-h] [-tr TRAIN] [-te TEST] [-m CLASSIFIER]
optional arguments:
  -h, --help            show this help message and exit
  -tr TRAIN, --train TRAIN
                        Your input train file.
  -te TEST, --test TEST
                        Your input test file.
  -m CLASSIFIER, --classifier CLASSIFIER
                        Machine learning classier.



## Stopping Condition
This method will find out the termination condition to stop the elimination of variables using feature selection’s cross-validation results 

**Note:**
The feature selection has to be done before implementing stop.py.

**Command**
python stop.py -tr Training filename -teTesting filename -m classifier -f feature selection
Command Example
python stop.py -tr UNSW_NB15_training-set.csv -te UNSW_NB15_testing-set.csv -m RF -f Union
python stop.py -tr UNSW_NB15_training-set.csv -te UNSW_NB15_testing-set.csv -m RF -f Greedy

Stop.py will print out the accuracy, F1-score, stopping features, stopping the number of features
Filename: CLASSIFIERNAME_METHOD_FILENAME.csv 

% python stop.py -h 
usage: stop.py [-h] [-tr TRAIN] [-te TEST] [-m CLASSIFIER] [-f FS]
optional arguments:
  -h, --help            show this help message and exit
  -tr TRAIN, --train TRAIN
                        Your input train file.
  -te TEST, --test TEST
                        Your input test file.
  -m CLASSIFIER, --classifier CLASSIFIER
                        Machine learning classier.
  -f FS, --fs FS        Feature selection methods.



