#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 19:07:48 2017

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

sys.setrecursionlimit(2500)
dataset_name = "MNIST"

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
   
    testX = mnist['testX'][:100,]
    testY = mnist['testY'][:100,]
    
    trainX = mnist['trainX'][:150,]
    trainY = mnist['trainY'][:150,]

dt = DT()
tree = dt.construct(trainX, trainY, 1)
y_hat = dt.predict(tree, testX[2])

print(y_hat)
