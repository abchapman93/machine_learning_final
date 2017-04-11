#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 08:37:49 2017

@author: alec
"""

'''
BMI 6950 - Applied Machine Learning

@author: jferraro
'''
import os
import sys
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from DrawDigits import draw_digits
from Tree import Tree
from KNN import KNN
from DT import DT

'''
    First we will read in some hand-written digits and
    take a look at the problem space we will be addressing.
    The data has been stored in a *.mat MATLAB format.
'''
DATADIR=os.getcwd()


dataset_name = 'MNIST'

'''
    Now let's load the training and test data that we plan 
    to try and classify. There are 10 classification labels,
    one for each digit. 
'''
if (dataset_name == 'MNIST'):
    mnistfile = os.path.join(DATADIR, 'mnist-test.mat')
    mnist_test = sio.loadmat(mnistfile)
    test_trainX = mnist_test['trainX']
    test_trainY = mnist_test['trainY']
    seq = np.arange(0,len(test_trainY),20)
    ttrX = test_trainX[seq,:]
    ttrY = test_trainY[seq]
    draw_digits(ttrX, ttrY)

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
    This is an implementation detail that needs to be
    addressed because we are using Python. We will be 
    using recursion to build our decision tree so we
    need to tell Python that we want plenty of stack
    space since we may recurse deep. 
'''
sys.setrecursionlimit(2500)

'''
    Now, we use cross-validation to select model parameters for 
    the K-Nearest Neighbor classifier. We will use 5-fold because 
    it is faster, but remember that you should really use 10-fold 
    in practice. We will evaluate K = 1, 3, 5, 10, 15, and 20
'''


'''
    Now, we use cross-validation to select model parameters for 
    the Decision Tree classifier. We will use 5-fold because 
    it is faster, but remember that you should really use 10-fold 
    in practice. We will evaluate K = 1, 3, 5, 10, 15, 20, 40
'''
dt = DT()
K = [1, 3, 5, 10, 15, 20, 40]

N = len(trainY)
results = []
fold = np.concatenate(np.random.randint(5, size=(N,1)))
for k in range(len(K)):
    print("DT K = " + str(K[k]))
    test_count = 0
    for f in range(0, 5):
        fold_count = 0
        trX = trainX[fold != f,:]
        trY = trainY[fold != f]
        tstX = trainX[fold == f,:]
        tstY = trainY[fold == f]
        tree = dt.construct(trX, trY, K[k])
        print(tree)

        for i in range(0, len(tstY)):
            y_hat = dt.predict(tree, tstX[i])
            if (y_hat == tstY[i]):
                fold_count += 1
                test_count += 1
        print("   Fold %d accuracy: %.2f %%" % ((f + 1), ((fold_count/len(tstY)) * 100.0)))                        
    print("     Overall test accuracy: %.2f %%" % ((test_count/N) * 100))  
    train_count = 0
    tree = dt.construct(trainX, trainY, K[k])
    for i in range(0, len(trainY)):
        y_hat = dt.predict(tree, trainX[i])
        if (y_hat == trainY[i]):
            train_count +=1
    print("     Overall training accuracy: %.2f %%" % ((train_count/N) * 100.0))  
    results.append([(train_count/N), (test_count/N)])       

'''
    Now, let's graph the test and training error so we can 
    evaluate the best model parameters.
'''
tracc = [item[0] for item in results]
tstacc = [item[1] for item in results]
plt.figure(3)
trainAcc = plt.plot(K, tracc, '-b', label='Training Error')
testACC = plt.plot(K, tstacc, '-g', label='Testing Error')
plt.title("DT Accuracy versus K Model Parameter")
plt.ylabel("Accuracy")
plt.xlabel("K")
plt.legend(loc=1)
plt.show(block=False)
print("Done!")

