'''
BMI 6950 - Applied Machine Learning

@author: jferraro
'''
import numpy as np
from numpy import  linalg as linalg
from collections import OrderedDict
import math
class KNN():
    '''
    K-Nearest Neighbor Classifier
    '''

    def predict(self, trainX, trainY, K, X):
        '''
            trainX - N x D matrix of training case features
            trainY - N x 1 matrix of training case labels
            K - regularization parameter
            X - 1 x D instance represented by features to predict on
            y - this is the predicted class that your are to
                return 
            This is where you will implement the KNN learner
        '''
        def euclid_dist(X, training_X):
            """Calculates the euclidean distance between X and every training instance
            Returns an ordered dictionary of the distance of instances and their indices"""
            distances = {}
            for i in range(len(training_X)):#is x-hat the training instance?
                instance = training_X[i]
                distance = math.sqrt((X-instance).dot(X-instance).T) #where would the sum come in?
                distances[i] = distance #the key-value pair is the index from training_X and the distance
            distances = OrderedDict(sorted(distances.items(),key=lambda t: t[1]))
            return distances
        distances = list(euclid_dist(X, trainX).items()) #this is a list of tuples
                                                        # [(index,distance),...]
        k_nearest = [x for x in distances[0:K]] #these are the k-nearest neighbors
        #print('k= %d'%K,k_nearest)
        k_labels = [] #identify the indices of these neighbors in the original 
        k_indices = [pair[0] for pair in k_nearest]
        k_labels = [trainY[i] for i in k_indices] #use those indices to get the labels from trainY
        
        y = int(k_labels[np.argmax(k_labels)])
        #print('Prediction:',y)
        
        return y

    