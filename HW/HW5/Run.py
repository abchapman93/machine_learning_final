'''
BMI 6950 - Applied Machine Learning

@author: jferraro
'''
import os
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.svm import LinearSVC


def plotgraph(figure, results, C):
    plt.figure(figure)
    for index, row in results.iterrows():
        name = row['name']
        acc = row['accuracy']
        tracc = [item[0] for item in acc]
        tstacc = [item[1] for item in acc]
        trainAcc = plt.plot(C, tracc, label=name + ' - Training Error', ls='dashed', linewidth=2)
        color = trainAcc[0].get_color()
        testACC = plt.plot(C, tstacc, label=name + ' - Testing Error', color=color, linewidth=2)
    params = {'legend.fontsize': 8}
    plt.rcParams.update(params)
    axis = plt.gca()
    axis.set_ylim([0.0, 1.05])
    plt.title("SVM Accuracy versus Slack")
    plt.ylabel("Accuracy")
    plt.xlabel("C")
    plt.legend(loc=4)
    plt.show(block=False)

print (sys.version)

'''
    First we will read in some hand-written digits and
    take a look at the problem space we will be addressing.
    The data has been stored in a *.mat MATLAB format.
'''

DATADIR=os.getcwd()
dataset_name = input("Enter dataset name <MNIST | 20NG>")

'''
    Now let's load the training and test data that we plan 
    to try and classify. There are 10 classification labels,
    one for each digit. 
'''
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
acc = []

slacks = [2, 1, 0.1, 0.05, 0.01]
for slack in slacks:
    print('Linear SVM, C=%s'%str(slack))
    test_count = 0
    for f in range(0, nfolds):
        fold_count = 0
        trX = trainX[fold != f,:]
        trY = trainY[fold != f]
        tstX = trainX[fold == f,:]
        tstY = trainY[fold == f]
        svc_clf = LinearSVC(C=float(slack),loss='hinge')
        svc_clf.fit(trX, np.ravel(trY))
        y_hat = svc_clf.predict(tstX)
        for i in range(len(y_hat)):
            if y_hat[i] == np.ravel(tstY)[i]:
                fold_count += 1
                test_count += 1
       # print("   Fold %d accuracy: %.2f %%" % ((f + 1), ((fold_count/len(tstY)) * 100.0)))
    test_accuracy = (test_count/N)
    print("     Overall test accuracy: ", test_accuracy) #((test_count/N) * 100))
    train_count = 0
    svc_clf = LinearSVC(C=float(slack),loss='hinge')
    svc_clf.fit(trainX, np.ravel(trainY))
    y_hat = svc_clf.predict(trainX)
    for i in range(len(y_hat)):
        if y_hat[i] == np.ravel(trainY)[i]:
            train_count += 1
    train_accuracy = (train_count/N)
    print("Overall training accuracy: ",train_accuracy)
    acc.append([(train_accuracy), (test_accuracy)])
results1 = results1.append({'name':'Lin SVM', 'accuracy': acc},ignore_index=True)
plotgraph(1, results1,slacks)
result = results1['accuracy']
for row in result:
    for test_train in row:
        with open(os.path.join(os.getcwd(),'results_linear_{0}.csv'.format(dataset_name)),'a') as f0:
            f0.write(str(test_train)+'\n')

#==============================================================================
# 
# Polynomial SVM
# 
#==============================================================================

from sklearn.svm import SVC
results2 = pd.DataFrame(columns=('name', 'accuracy'))



degrees = [2, 3]
slacks = [6, 5, 3, 1, 0.5, 0.1]
coeffs = [10, 2, 1]


for d in degrees:
    for coef in coeffs:
        acc = []
        model_name = 'SVM Poly = {0} Coef = {1}'.format(d, coef)
        for slack in slacks:
            
            #print(model_name)
            test_count = 0
            for f in range(0, nfolds):
                fold_count = 0
                trX = trainX[fold != f,:]
                trY = trainY[fold != f]
                tstX = trainX[fold == f,:]
                tstY = trainY[fold == f]
                svc_clf = SVC(C=slack, kernel='poly', degree=d, coef0=coef)
                svc_clf.fit(trX, np.ravel(trY))
                y_hat = svc_clf.predict(tstX)
                for i in range(len(y_hat)):
                    if y_hat[i] == np.ravel(tstY)[i]:
                        fold_count += 1
                        test_count += 1
                #print("   Fold %d accuracy: %.2f %%" % ((f + 1), ((fold_count/len(tstY)) * 100.0)))
            test_accuracy = (test_count/N)
            #print("     Overall test accuracy: ", test_accuracy) #((test_count/N) * 100))
            train_count = 0
            svc_clf = SVC(C=slack, kernel='poly', degree=d, coef0=coef)
            svc_clf.fit(trainX, np.ravel(trainY))
            y_hat = svc_clf.predict(trainX)
            for i in range(len(y_hat)):
                if y_hat[i] == np.ravel(trainY)[i]:
                    train_count += 1
            train_accuracy = (train_count/N)
            #print("Overall training accuracy: ",train_accuracy)
            acc.append([(train_accuracy), (test_accuracy)])
        results2 = results2.append({'name':model_name, 'accuracy': acc},ignore_index=True)

plotgraph(2, results2,slacks)

result = results2['accuracy']
for row in result:
    for train_test in row:
        with open(os.path.join(os.getcwd(),'results_poly_{0}.csv'.format(dataset_name)),'a')as f0:
            f0.write(str(train_test)+'\n')




#==============================================================================
# RBF
#==============================================================================
from sklearn.svm import SVC
results3 = pd.DataFrame(columns=('name', 'accuracy'))




slacks = [25, 20, 10, 1, 0.1, 0.01]
gammas = [0.0001,0.005, 0.01, 0.05, 0.1]


for gamma in gammas:
    acc = []
    model_name = 'SVM RBF Gamma={0}'.format(gamma)
    for slack in slacks:
        #print(model_name)
        test_count = 0
        for f in range(0, nfolds):
            fold_count = 0
            trX = trainX[fold != f,:]
            trY = trainY[fold != f]
            tstX = trainX[fold == f,:]
            tstY = trainY[fold == f]
            svc_clf = SVC(C=slack, kernel='rbf', gamma=gamma)
            svc_clf.fit(trX, np.ravel(trY))
            y_hat = svc_clf.predict(tstX)
            for i in range(len(y_hat)):
                if y_hat[i] == np.ravel(tstY)[i]:
                    fold_count += 1
                    test_count += 1
            #print("   Fold %d accuracy: %.2f %%" % ((f + 1), ((fold_count/len(tstY)) * 100.0)))
        test_accuracy = (test_count/N)
        #print("     Overall test accuracy: ", test_accuracy) #((test_count/N) * 100))
        train_count = 0
        svc_clf = SVC(C=slack,kernel='rbf',gamma=gamma)
        svc_clf.fit(trainX, np.ravel(trainY))
        y_hat = svc_clf.predict(trainX)
        for i in range(len(y_hat)):
            if y_hat[i] == np.ravel(trainY)[i]:
                train_count += 1
        train_accuracy = (train_count/N)
        #print("Overall training accuracy: ",train_accuracy)
        acc.append([(train_accuracy), (test_accuracy)])
    results3 = results3.append({'name':model_name, 'accuracy': acc},ignore_index=True)

plotgraph(3, results3,slacks)

result = results3['accuracy']
for row in result:
    for test_train in row:
        with open(os.path.join(os.getcwd(),'results_rbf_{0}.csv'.format(dataset_name)),'a')as f0:
            f0.write(str(test_train)+'\n')  



'''
This code will create your folds for testing each
one of your classifiers. Cut and paste it whereever you 
like to integrate it into your implementations.

'''


print("Done!")


all_results=pd.DataFrame()
all_results = all_results.append([results1,results2,results3])
all_results.to_csv(os.path.join(os.getcwd(),'all_results{0}.csv'.format(dataset_name)))