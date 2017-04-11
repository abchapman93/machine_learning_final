#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 19:48:08 2017

@author: alec
"""

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

DATADIR=os.getcwd()


#dataset_name = input("Enter dataset name <MNIST | 20NG>")
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

dt = DT()
K = [1, 3, 5, 10, 15, 20, 40]
N = len(trainY)
results = []
fold = np.concatenate(np.random.randint(5, size=(N,1)))
for k in range(2):
    print("DT K = " + str(K[k]))
    test_count = 0
    for f in range(0, 2): #should be 5
        fold_count = 0
        trX = trainX[fold != f,:]
        trY = trainY[fold != f]
        tstX = trainX[fold == f,:]
        tstY = trainY[fold == f]
        tree = dt.construct(trX, trY, K[k])
"""      
        for i in range(0, len(tstY)):
            y_hat = dt.predict(tree, tstX[i])
            if (y_hat == tstY[i]):
                fold_count += 1
                test_count += 1
        print("   Fold %d accuracy: %.2f %%" % ((f + 1), ((fold_count/len(tstY)) * 100.0)))                        
    print("     Overall test accuracy: %.2f %%" % ((test_count/N) * 100))  
    

"""