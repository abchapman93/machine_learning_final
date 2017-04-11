'''
BMI 6950 - Applied Machine Learning

@author: jferraro
'''
import math
import numpy.matlib as ml
from scipy import ndimage
import matplotlib.pyplot as plt

'''
    This is simply a utility function that will produce a
    picture of the hand-written digits and their corresponding 
    classes.
'''
def draw_digits(X,Y, figure_no=1):
    N, D = X.shape
    DY = math.floor(math.sqrt(N));
    DX = math.ceil(N / DY);
    plt.figure(figure_no)
    for n in range(0, N):
        plt.suptitle("K = " + str(N) + " Clusters")
        plt.subplot(DY, DX, n+1)
        Z = ml.uint8(255*X[n,:].reshape(28,28).T)
        im = plt.imshow(ndimage.rotate(Z, 90), interpolation='bilinear', origin='lower')
        plt.axis('off')    
        plt.title('C = ' + str(Y[n]))
    plt.show(block=False)   

