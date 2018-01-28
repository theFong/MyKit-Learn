from __future__ import division, print_function

from typing import List

import numpy
import scipy
import sklearn


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
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


class LinearRegressionWithL2Loss:
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
    print(sklearn.__version__)
