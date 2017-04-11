#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:19:39 2017

@author: alec
"""
import math
import numpy as np
from numpy import random as rnd
from collections import defaultdict
import DT, Tree, KNN
#X1 = rnd.randint(0,high=2,size=(10,5))
#Y1 = rnd.randint(0,high=2,size=(10,1))
X1 = np.array([[1,1,1,0],
              [1,0,1,1],
              [1,0,0,0],
              [0,1,1,0]])
Y1 = np.array([[0],
              [1],
              [1],
              [1]])

X2 = np.array([[1,1,1,0],
              [0,0,1,1],
              [1,0,1,0],
              [0,1,0,0]])
Y2 = np.array([[0],
              [0],
              [0],
              [1]])
#Y = list(np.squeeze(Y))
#Y2 = rnd.randint(0,high=9,size=(10,5))
#Y2 = [1,1,1,1,1,2,3,4,5,1,1,1,1,1,2]
Y2=np.zeros((10,1),dtype=int)
Y3 = np.ones((10,1),dtype=int)

#print(entropy(Y1))
test_tree = DT.DT()

final_tree = test_tree.construct(X1, Y1, 5)
#print(np.shape(X1))

print('DT prediction:',test_tree.predict(final_tree,X1))

knn = KNN.KNN()
print('knn prediction:',knn.predict(X1,Y1,3,[1,1,1,0]))