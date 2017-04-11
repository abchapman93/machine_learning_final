'''
BMI 6950 - Applied Machine Learning

@author: jferraro
'''
import math
import numpy as np
from Tree import Tree
import os
import sys

class DT(object):
    '''
    Decision Tree Classifier
    '''

    def __init__(self, split_threshold=0.5):
        '''
            No work to be done here. But you will use 
            the split_threshold in your construct and predict methods.
        '''
        self.split_threshold = split_threshold
          
    
    def __entropy(self, Y):
        ''' This is meant to be a private method as it 
            will only be called internally by your construct
            method. This is where you will implement 
            your purity function. 
            Y - N x 1 matrix representing class labels of the features in a node
                after a split
            entropy - you will return the entropy value 
            you calculate
        '''
        unique_counts = {} #this is the dictionary representing all of the classes present in Y
                        #unique_counts[label] = number of instances that has this label
        if len(Y) == 0:
            ent = 0
        else:
            for row in Y:
                label = int(np.squeeze(row))
                if label not in unique_counts:
                    unique_counts[label] = 0
                unique_counts[label] += 1
            m = len(Y)
            ent = 0
            for class_type in unique_counts:
                if unique_counts == 0:
                    pass
                else:
                    ent += - unique_counts[class_type]/m * math.log2(unique_counts[class_type]/m)
        return ent
        


    def construct(self, X, Y, threshold):
        '''
            X - N x D matrix of training case features
            Y - N x 1 matrix of training case labels
            threshold - regularization parameter controlling how
                        deep your tree should grow. 
            tree - this method should return your decision tree as
                   an instance of a Tree()
            This is where you will implement the Decision Tree learning
            method using recursion.
        '''
        tree = Tree()
        m = len(X)
        unique_counts = {} #this dictionary will represent all of the classes present in this node
        for row in range(len(Y)):
            instance_label = int(Y[row]) 
            #print('instance_label:',instance_label)
            if instance_label not in unique_counts:
                unique_counts[instance_label] = 0
            unique_counts[instance_label] += 1 #this is essentially the makeup of the node
        
        if len(unique_counts) == 1 or len(X) <= threshold: #if there is only one class in the node or if we've passed the threshold
            tree.isLeaf = 1
            v = list(unique_counts.values())
            k = list(unique_counts.keys())
            majority = k[v.index(max(v))] #this is the dictionary key with the largest value
            tree.label = majority
            return tree
        else:
            tree.isLeaf = 0
            
            possible_gains = []
            for column_index in range(np.shape(X)[1]): #iterate through the columns
                nodes = self.split_on_feature(X,Y,column_index,self.split_threshold)
                possible_gains.append(sum([(len(x)/m)*DT.__entropy(self,x) for x in nodes]))#calculate the combined entropy for the split
            best_gain = min(possible_gains)
            #this is the index of the feature that returned the best gain
            best_feature_index = possible_gains.index(best_gain)
            #now we can split the tree with this feature
            tree.split = best_feature_index
            true_node_X = []
            true_node_Y = []
            false_node_X = []
            false_node_Y = []
            for row in range(len(X)):
                if X[row,best_feature_index] >= self.split_threshold:
                    true_node_X.append(X[row])
                    true_node_Y.append(Y[row])
                else:
                    false_node_X.append(X[row])
                    false_node_Y.append(Y[row])
            X_S0 = np.array(false_node_X)
            Y_S0 = np.array(false_node_Y)
            X_S1 = np.array(true_node_X)
            Y_S1 = np.array(true_node_Y)
            if len(X_S1) != 0 and len(Y_S1) != 0 and (len(X_S0) != 0 and len(Y_S0) != 0):
                tree.right = DT.construct(self,X_S1,Y_S1,threshold)
                tree.left = DT.construct(self,X_S0,Y_S0,threshold)
            else:
                tree.isLeaf = 1
            return tree
            
            
    def predict(self, tree, X):
        '''
            tree - a trained decision tree model
            X - 1 x D instance represented by features to predict on
            This is where you will implement the Decision Tree prediction
            method, again using recursion.
        '''
        if tree.isLeaf:
            return tree.label
        else:
            if X[tree.split] <= self.split_threshold:
                return DT.predict(self,tree.left,X)
            else:
                return DT.predict(self,tree.right,X)

    def split_on_feature(self,X, Y,column,value):
        """Splits a matrix of labels up based on a given feature index (column) and a critera (value)
        Returns a list of arrays that represent all the nodes of this split that can be used to calculate information gain"""
        yes_node = [] 
        no_node = []
        for i in range(len(X)): #i = row in X.
            if X[i,column] >= value: #if this is a new value of the given feature
                yes_node.append(Y[i])
            else:
                no_node.append(Y[i])
        return yes_node, no_node # this is a list representing all the new nodes
