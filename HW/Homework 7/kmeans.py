'''
Created on Mar 27, 2017

@author: jferraro
'''
import numpy as np
from numpy import linalg as la
from scipy.spatial.distance import sqeuclidean

def kmeans(X, K, init):
    '''
    input X is N x D data, K is number of clusters desired.  init is either
    'random' which just chooses K random points to initialize with, or
    'furthest', which chooses the first point randomly and then uses the
    "furthest point" heuristic.  The output is mu, which is K x D, the
    coordinates of the means, and z, which is N x 1, and only integers 1...K
    specifying the cluster associated with each data point.  Score is the
    score of the clustering
    '''
    N, D = X.shape

    if init == 'random': #change back to ==
        '''
        Initialize by choosing K (distinct!) random points: we do this by
        randomly permuting the examples and then selecting the first K
        '''
        perm = np.random.permutation(N)
        perm = perm[0:K]
        mu = X[perm, :]
    else:
        '''
        Initialize the first center by choosing a point at random; then
        iteratively choose furthest points as the remaining centers

        Randomly choose the first center c_1.
        Then, let c_2 = argmax dist(x_i, c_1).
        Let  c_3 = argmax min [ dist(x_i, c_1), dist(x_i, c_2) ]
        and so on....
        '''

        '''

        % TODO  - Implement furthest initialization here

        mu = {initial centroids using furthest initialization ...}

        '''
        mu = []
        c_1_index = np.random.choice(N)
        c_1 = X[[c_1_index],:]
        mu.append(c_1)

        while len(mu) != K:
            min_distances = []
            for i in range(N): #for each data point
                x = X[[i],:]
                x_distances = [] #a list of distances between data point x and each centroid
                for c in mu: #for each existing centroid
                    dist = sqeuclidean(x,c)#scipy.spatial.distance.euclidean
                    x_distances.append(dist)
                x_min_index = np.argmin(x_distances) #returns the index of the minumum distance between the data point and c
                min_distances.append((i,x_distances[x_min_index])) #append the minimum distance to min_distances
            min_indices, min_values = zip(*min_distances)
            new_c_index = np.argmax(min_values) #select the maximum min_value; this is an index
            new_c = X[[new_c_index],:]
            mu.append(new_c)

    '''
        We leave the assignments unspecified -- they'll be computed in the
        first iteration of K means
    '''
    z = np.zeros((N,1), dtype=int) #one column for each data point

    '''
        Begin the iterations.  We'll run for a maximum of 20, even though we
        know that things will *eventually* converge.

    '''
    iter_count = 0
    for iter in range(20):
        '''
            In the first step, we do assignments: each point is assigned to the
            closest center.  We'll judge convergence based on these assignments,
            so we want to keep track of the previous assignment
        '''
        oldz = np.array(z)
        for n in range(N): #each data point

            '''
            % TODO - Assign point n to the closest centroid (i.e. closest cluster) ...

            k = {assigned cluster for point n . . .}

            '''
            x= X[[n],:] #values of this data point
            x_distances = [sqeuclidean(x,c) for c in mu] #distances between x and each existing centroid
            k = np.argmin(x_distances) #the index of the closer centroid

            z[n] = k #the cluster name is the index in mu


        ''' check to see if we've converged '''
        if (np.all(oldz == z)):
            break;

        else:
            '''
                re-estimate the means mu
                % TODO
            '''
            iter_count += 1
            new_mu = []
            for index in range(K):
                subset = [] #create a subset of points that are clustered together
                for i in range(N):
                    if np.squeeze(z)[i] == index:
                        subset.append(X[[i],:])
                new_c = np.mean(subset,axis=0)
                new_mu.append(new_c) #average all the points, this becomes the new centroid
            mu = new_mu



    ''' final: compute the score '''
    score = 0;
    mu = np.array(mu)
    mu = np.squeeze(mu,axis=(1,))
    for n in range(N):
        score = score + la.norm(X[n] - mu[z[n]])**2
        z1 = la.norm(X[n] - mu[z[n]])**2;
        z2 = 0;
        for i in range(X.shape[1]):
            z2 = z2 + (X[n, i] - mu[z[n], i])**2
    return(mu, z, score)
