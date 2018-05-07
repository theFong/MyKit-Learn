from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        # number of classes
        self.nb_features = 2
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        ############################################################################
        # This should take a list of features and labels [-1,1] and should update 
        # to correct weights w. Note that w[0] is the bias term. and first term is 
        # expected to be 1 --- accounting for the bias
        ############################################################################
        # iterate max_iteration times
        is_terminate = False
        for i in range(0,self.max_iteration):
            # iterate over all features
            if is_terminate:
                return True
            is_terminate = True
            for x,y in zip(features, labels):
                x = np.array(x)
                pred = np.array(self.w).transpose().dot(x)
                # cast to -1,1
                pred_sign = np.sign(pred)
                if pred_sign != y:
                    # if predicted label is wrong
                    # update rule -> Wnew = W + YX/|X|
                    x_norm = np.linalg.norm(x)
                    self.w = np.array(self.w) + ((y * x) / (x_norm + self.margin))
                    is_terminate = False

        return False     
    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        ############################################################################
        # This should take a list of features and use the learned 
        # weights to predict the label
        ############################################################################
        return np.sign(np.inner(self.w.transpose(),features))

    def get_weights(self) -> Tuple[List[float], float]:
        return self.w
    