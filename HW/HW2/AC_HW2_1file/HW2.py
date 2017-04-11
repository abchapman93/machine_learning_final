#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:52:57 2017

@author: alec
"""

import os


import matplotlib
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
DATADIR = os.path.join(os.getcwd(),'Homework 2 Resources')
os.path.exists(DATADIR)
from numpy import random as rnd

fFeatures = os.path.join(DATADIR, 'X.txt')
fLabels = os.path.join(DATADIR, 'y.txt')

with open(fFeatures,'r') as f0:
    X = np.array([float(x.strip()) for x in f0.readlines()])
with open(fLabels,'r') as f1:
    y = np.array([float(x.strip()) for x in f1.readlines()])
X_new = np.c_[np.ones((len(X),1)),X]
np.shape(X_new)



learning_rates = [0.0001, 0.01, 0.5]

def gradient_descent(X, y, eta, n_iterations, theta=rnd.randn(2,1)):
    
    for iteration in range(n_iterations):
        gradients = 2/(len(X)) * X.T.dot(X.dot(theta)-y)
        theta = theta - eta*gradients
    return theta
    

#Normal Equation
theta_best = LA.inv(X_new.T.dot(X_new)).dot(X_new.T).dot(y)
y_predict = X_new.dot(theta_best)
plt.plot(X_new,y_predict,"r_", linewidth=2, label="Predictions")
plt.plot(X, y, "b.")
plt.xlabel("$x$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 2, 0, 15])
plt.show()
#plt.scatter(X_new,theta_best)

#thetas = gradient_descent(X_new, y, learning_rates[2],2000)

#y_predict = X_new.dot(thetas)
"""
#Report model parameters for the first 5 descent iterations and final iteration
#Graph the state of the hypothesis every 200th iteration

def plot_grad_descent(X, y, eta, n_iterations=2001, theta=rnd.randn(2,1)):
    for iteration in range(n_iterations):    
        
        gradients = 2/len(X) * X.T.dot(X.dot(theta)-y)
        theta = theta - eta*gradients
        
        #if iteration <= 5.0:
            #first_five_thetas.append(theta)
        if iteration == 2000:
            print(np.shape(theta))
            print(iteration)
            plt.scatter(X,thetas.T)
            plt.show()
            return
    #print('First five:',first_five_thetas)
    #print('Every 200th:', iterations_200)
    return 
    
plot_grad_descent(X_new, y, learning_rates[0])
    
"""