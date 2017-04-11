import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def logistic_func(theta, x):
    #    theta - matrix of model parameters
    #    x - matrix of training case features
    #
    # define the sigmoid function here....
    
def cost_func(theta, x, y):
    #    theta - matrix of model parameters
    #    x - matrix of training case features
    #    y - matrix of training case labels
    #
    # define the cost function J(theta) here ....
    
def pred_values(theta, X):
    #    theta - matrix of model parameters
    #    x - matrix of training case features
    #
    #    theta should contain your trained model parameters.
    #    You just need to call your hypothesis function here
    #    Remember your hypothosis functin returns a probability
    #    that needs to be mapped into a classification. That should
    #    not be too hard. 

# Load your classification data here
#X = ???
#y = ???
# (Hint: check the shape of your data structures. You can
#  reshape things with if need be.

# Plot your data features color coded by class here...
class1 = plt.scatter(..., c='r')
class2 = plt.scatter(...,, c='b')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend((class1, class2), ("Class 1", "Class 0"), loc=2)

# You may want to augment you feature matrices here if you are 
# planning to implement your gradient descent using linear algebra

cost_threshold = ?
change_cost = ?
#record your initial descent cost (i.e. cost_func() )
while(change_cost > cost_threshold):
    # implement gradient descent here
    # track your descent cost here
    # change_cost = old_cost - new_cost

# predict the performance of your learned model (i.e. pred_values() )
# ...

# now plot your cost per iteration here
plt.plot(iteration, cost)
plt.ylabel("Cost")
plt.xlabel("Iteration")


# Now implement the Scikit-Learn Logistic Regression function here . . .

print("Done!")


