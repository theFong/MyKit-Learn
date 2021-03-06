from __future__ import division, print_function

from typing import List

import numpy
import scipy

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        # creating f(x) = W0 + sum((Wd)(Xd))
        # finding W's
        # use LMS(least mean squared(minimizing residuals sum(mean squared error)))
        # LMS = ((X^tX)^-1) (X^tY) 
        # make sure N > D+1
        # features = X values = y
        y = numpy.array([values]).transpose()
        x = numpy.array(features)

        # add w0
        x = numpy.append([[1]]*len(features),x, axis=1)

        xtx = x.transpose().dot(x)
        xty = x.transpose().dot(y)
        xtxInv = numpy.linalg.inv(xtx)
        self.weights = xtxInv.dot(xty)
       
    def phi(self, x: List[float]) -> List[float]:
        aug = []
        for i in range(1,len(x)):
            for k in range(2,self.nb_features+1):
                aug.append(numpy.power(x[i],k))
        return aug

    def predict(self, features: List[List[float]]) -> List[float]:
        # f(x) = wtx
        # x -> p(x) [1,x,x^2...x^d]
        x = numpy.array(features)
        x = numpy.append([[1]]*len(features),x, axis=1)
        
        return numpy.inner(self.weights.transpose(),x)[0]

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
        # creating f(x) = W0 + sum((Wd)(Xd))
        # finding W's
        # use LMS(least mean squared(minimizing residuals sum(mean squared error)))
        # LMS = ((X^tX)^-1) (X^tY) Now (X^tX -> X^tX + lI)
        # make sure N > D+1
        # features = X values = y
        y = numpy.array([values]).transpose()
        x = numpy.array(features)

        # add w0
        x = numpy.append([[1]]*len(features),x, axis=1)

        # xtx -> xtx + lI
        xtx = numpy.add(x.transpose().dot(x), self.alpha * numpy.identity(x.shape[1]))
        xty = x.transpose().dot(y)
        xtxInv = numpy.linalg.inv(xtx)
        self.weights = xtxInv.dot(xty)

    def phi(self, x: List[float]) -> List[float]:
        aug = []
        for i in range(1,len(x)):
            for k in range(2,self.nb_features+1):
                aug.append(numpy.power(x[i],k))
        return aug

    def predict(self, features: List[List[float]]) -> List[float]:
        # f(x) = wtx + l|w|^2
        # x -> p(x) [1,x,x^2...x^d]
        x = numpy.array(features)
        x = numpy.append([[1]]*len(features),x, axis=1)
        
        return numpy.inner(self.weights.transpose(),x)[0]

    def get_weights(self) -> List[float]:
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.weights


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
