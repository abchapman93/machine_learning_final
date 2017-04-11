'''
BMI 6950 - Applied Machine Learning

@author: jferraro
'''

class Tree():
    '''
    This class represents a decision tree. 
    isLeaf - Tells us if this node is a branch(=0) or a leaf(=1)
    label - When the node is a leaf node the label signifies the class that is retruned
    split - When the node is a branch, this field stores the feature to use for 
            sending an instance down the left or right branch
    left - When a node is a branch, this represents the left subtree - it is
            a tree itself.
    right - When a node is a branch, this represents the right subtree - it is
            a tree itself            
    '''
    def __init__(self, isLeaf=None, label=None, split=None, left=None, right=None):
        self.isLeaf = isLeaf
        self.label = label
        self.split = split
        self.left = left
        self.right = right
        

