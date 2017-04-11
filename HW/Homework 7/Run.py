'''
BMI 6950 - Applied Machine Learning

@author: jferraro
s'''
import os
import sys
import math
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from DrawDigits import draw_digits
from kmeans import kmeans

def plotgraph(figure, results, K):
    plt.figure(figure)
    for index, row in results.iterrows():
        name = row['name']
        scores = row['score']
        marker = row['marker']
        scoreplot = plt.plot(K, scores, label=name, marker=marker, linewidth=1)
    params = {'legend.fontsize': 8}
    plt.rcParams.update(params)
    axis = plt.gca()
    axis.set_ylim([0.65, 1.1])
    plt.title("K versus score")
    plt.ylabel("Score")
    plt.xlabel("K")
    plt.legend(loc=3) 
    plt.show(block=False)


print (sys.version)

DATADIR=r'.'

'''
    Now let's load the training and test data that we plan 
    to try and classify. There are 10 classification labels,
    one for each digit. 
'''
mnistfile = os.path.join(DATADIR, 'mnist.mat')
mnist = sio.loadmat(mnistfile)
trainX = mnist['trainX']

N, D = trainX.shape
allK = [2, 5, 10, 15, 20, 25]
scores = np.zeros(len(allK))
results = pd.DataFrame(columns=('K', 'clusterAssignments', 'clustermeans'))
for ii in range(len(allK)):
    print("K=%d..." % (allK[ii])) 
    bestScore = float("inf")
    bestZ = 0
    bestMu = 0
    for rep in range(3):
        mu, z, s = kmeans(trainX, allK[ii], 'furthest')
        if (s < bestScore):
            bestScore = s
            bestZ = z
            bestMu = mu
    print(" --> score %f" % (bestScore))                            
    scores[ii] = bestScore
    results = results.append({'K' : allK[ii], 'clusterAssignments': bestZ, 'clustermeans' : bestMu}, ignore_index=True)      

figure = 1
for index, row in results.iterrows():
    K = int(row['K'])
    clusterAssignments = row['clusterAssignments']
    clustermeans = row['clustermeans']
    k = np.asarray(list(range(K))).reshape(K,1) + 1
    draw_digits(clustermeans, k, figure)
    figure += 1

bicscores = np.zeros(len(allK))
aicscores = np.zeros(len(allK))
for ii in range(len(allK)):
    bicscores[ii] = N * np.log(scores[ii] / N) + allK[ii] * np.log(N);
    aicscores[ii] = N * (np.log(2 * math.pi * scores[ii] / N) + 1) + 2 * allK[ii];
    
scores = scores / max(scores)
bicscores = bicscores / max(bicscores)
aicscores = aicscores / max(aicscores)  

results = pd.DataFrame(columns=('name', 'score', 'marker'))
results = results.append({'name': 'Normalized Score', 'score' : scores, 'marker' : 'D'}, ignore_index=True)      
results = results.append({'name': 'Normalized BIC', 'score' : bicscores, 'marker' : 'o'}, ignore_index=True)      
results = results.append({'name': 'Normalized AIC', 'score' : aicscores, 'marker' : 'p'}, ignore_index=True)      
plotgraph(figure, results, allK)

print("Done!")

