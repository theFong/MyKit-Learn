from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        # creating f(x) = W0 + sum((Wd)(Xd))
        # finding W's
        # use LMS(least mean squared(minimizing residuals sum(mean squared error)))
        # LMS = ((X^tX)^-1) (X^tY) (X^tX -> X^tX + lI)
        # make sure N > D+1
        # features ~ X values ~ y
        assert self.nb_features == len(features)
        y = numpy.array([values]).transpose()
        x = numpy.array(features)
        # to add w0
        x = numpy.append(x, [[1]]*len(features), axis=1)

        xtx = x.transpose().dot(x)
        xty = x.transpose().dot(y)
        xtxInv = numpy.linalg.inv(xtx)
        self.weights = xtxInv.dot(xty)
        

    def predict(self, features: List[List[float]]) -> List[float]:
        # f(x) = wx + w0
        return numpy.add(numpy.dot(self.weights,features.transpose()), self.weights[1][0])[0]

    def get_weights(self) -> List[float]:
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.weights

class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        raise NotImplementedError

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        raise NotImplementedError


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
