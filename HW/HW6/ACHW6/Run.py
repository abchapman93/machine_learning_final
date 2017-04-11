#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 09:07:45 2017

@author: alec
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plotgraph(figure, results, C):
    plt.figure(figure)
    for index, row in results.iterrows():
        name = row['name']
        acc = row['accuracy']
        tracc = [item[0] for item in acc]
        tstacc = [item[1] for item in acc]
        trainAcc = plt.plot(C, tracc, ls='dashed', linewidth=2)#label=name + ' - Training Error',
        color = trainAcc[0].get_color()
        testACC = plt.plot(C, tstacc, label=name + ' - Testing Error', color=color, linewidth=2)
    params = {'legend.fontsize': 8}
    plt.rcParams.update(params)
    axis = plt.gca()
    axis.set_ylim([0.0, 1.05])
    plt.title("Random Forest Accuracy vs. n_estimators")
    plt.ylabel("Accuracy")
    plt.xlabel("n_estimators")
    plt.legend(loc=4)
    plt.show(block=False)
    
DATADIR = os.getcwd()
#dataset_name = input("Enter dataset name <MNIST | 20NG>")
dataset_name = "20NG"

if (dataset_name == 'MNIST'):
    mnistfile = os.path.join(DATADIR, 'mnist.mat')
    mnist = sio.loadmat(mnistfile)
    testX = mnist['testX']
    testY = mnist['testY']
    trainX = mnist['trainX']
    trainY = mnist['trainY']
elif (dataset_name == '20NG'):
    ng20file = os.path.join(DATADIR, '20ng-py.mat') 
    ng20 = sio.loadmat(ng20file)
    trainX = ng20['trainX']
    trainY = ng20['trainY']
else:
    print("Invalid dataset name")
    sys.exit()
    

'''
    Now, we use cross-validation to select hyper-parameters for 
    the SVM classifiers. We will use 5-fold because 
    it is faster, but remember that you should really use 10-fold 
    in practice. 
''' 

N = len(trainY)
results = []
nfolds = 5
fold = np.concatenate(np.random.randint(nfolds, size=(N,1)))

results1 = pd.DataFrame(columns=('name', 'accuracy'))


n_estimators = [5,25, 50,75,100,200] #number of learners  200
max_features = [10,25,50]#,"sqrt"
max_depth = [10,50]#None
#impurity_split = [10**-7, 2*(10**-7)]

#for split in impurity_split:
#f1 = open(os.path.join(os.getcwd(),'results_forest_{0}.csv'.format(dataset_name)),'a')
with open(os.path.join(os.getcwd(),'results_forest_{0}.csv'.format(dataset_name)),'w') as f0:
                writer = csv.writer(f0,delimiter=',')
                writer.writerow(['max_features','max_depth','n_estimators','train accuracy','test accuracy'])
for feat in max_features:
    for depth in max_depth:
        acc = []
        model_name = "max_features = {0},max_depth = {1}".format(feat, depth)
        for n in n_estimators:
            test_count = 0
            for f in range(0, nfolds):
                fold_count = 0
                trX = trainX[fold != f,:]
                trY = trainY[fold != f]
                tstX = trainX[fold == f,:]
                tstY = trainY[fold == f]
                rnd_clf = RandomForestClassifier(n_estimators=n, max_features=feat,
                                                 max_depth=depth)
                rnd_clf.fit(trX, np.ravel(trY))
                y_hat = rnd_clf.predict(tstX)
                for i in range(len(y_hat)):
                    if y_hat[i] == np.ravel(tstY)[i]:
                        fold_count += 1
                        test_count += 1
            test_accuracy = (test_count/N)
            print('model name:',model_name,n)
            print("     Overall test accuracy: ", test_accuracy)
            train_count = 0
            rnd_clf = RandomForestClassifier(n_estimators=n, max_features=feat,
                                                 max_depth=depth)
            rnd_clf.fit(trX, np.ravel(trY))
            y_hat = rnd_clf.predict(trainX)
            for i in range(len(y_hat)):
                if y_hat[i] == np.ravel(trainY)[i]:
                    train_count += 1
            train_accuracy = (train_count/N)
            print("Overall training accuracy: ",train_accuracy)
            acc.append([(train_accuracy), (test_accuracy)])
            with open(os.path.join(os.getcwd(),'results_forest_{0}.csv'.format(dataset_name)),'a') as f1:
                writer = csv.writer(f1,delimiter=',')
                writer.writerow([str(feat),str(depth),str(n),str(train_accuracy),str(test_accuracy)])
            #f1.writelines([model_name,',',str(n),',',])
            #f1.write(str(train_accuracy)+','+str(test_accuracy)+'\n')
        results1 = results1.append({'name':model_name, 'accuracy':acc},ignore_index=True)
plotgraph(0, results1, n_estimators)
##Figure out the indentation of the results1.append statement
#results1.to_csv(os.path.join(os.getcwd(),'results_forest_{0}all_2.csv'.format(dataset_name)))

names = results1['name']
result = results1['accuracy']

#==============================================================================
# AdaBoost
#==============================================================================
                
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
N = len(trainY)
results = []
nfolds = 5
fold = np.concatenate(np.random.randint(nfolds, size=(N,1)))

results2 = pd.DataFrame(columns=('name', 'accuracy'))


n_estimators = [500,750,1000] #number of learners  200
#max_features = [10,25,50]#,"sqrt"
learning_rates = [0.25,0.5,0.75]

with open(os.path.join(os.getcwd(),'results_ada_{0}.csv'.format(dataset_name)),'w') as f0:
                writer = csv.writer(f0,delimiter=',')
                writer.writerow(['learning rate','n_estimators','train accuracy','test accuracy'])
#for feat in max_features:
for rate in learning_rates:
    acc = []
    model_name = "learning rate={0}".format(rate)
    for n in n_estimators:
        test_count = 0
        for f in range(0, nfolds):
            fold_count = 0
            trX = trainX[fold != f,:]
            trY = trainY[fold != f]
            tstX = trainX[fold == f,:]
            tstY = trainY[fold == f]
            ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                n_estimators=n,learning_rate=rate)
            ada_clf.fit(trX, np.ravel(trY))
            y_hat = ada_clf.predict(tstX)
            for i in range(len(y_hat)):
                if y_hat[i] == np.ravel(tstY)[i]:
                    fold_count += 1
                    test_count += 1
        test_accuracy = (test_count/N)
        print('model name:',model_name,n)
        print("     Overall test accuracy: ", test_accuracy)
        train_count = 0
        ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                n_estimators=n,learning_rate=rate)
        ada_clf.fit(trX, np.ravel(trY))
        y_hat = ada_clf.predict(trainX)
        for i in range(len(y_hat)):
            if y_hat[i] == np.ravel(trainY)[i]:
                train_count += 1
        train_accuracy = (train_count/N)
        print("     Overall train accuracy: ",train_accuracy)
        acc.append([(train_accuracy), (test_accuracy)])
        with open(os.path.join(os.getcwd(),'results_ada_{0}.csv'.format(dataset_name)),'a') as f1:
            writer = csv.writer(f1,delimiter=',')
            writer.writerow([str(rate),str(n),str(train_accuracy),str(test_accuracy)])
        #f1.writelines([model_name,',',str(n),',',])
        #f1.write(str(train_accuracy)+','+str(test_accuracy)+'\n')
    results2 = results2.append({'name':model_name, 'accuracy':acc},ignore_index=True)
plotgraph(0, results2, n_estimators)
##Figure out the indentation of the results1.append statement
#results1.to_csv(os.path.join(os.getcwd(),'results_forest_{0}all_2.csv'.format(dataset_name)))

