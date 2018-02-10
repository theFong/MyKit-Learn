from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    # sum (( actual - predicted ) ^2) /n
    totalError = 0.0
    for i in range(0, len(y_pred)):
        totalError += np.power((y_true[i] - y_pred[i]), 2)
    return totalError / len(y_pred)


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)

    raise NotImplementedError


def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:
    raise NotImplementedError


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    # l2 norm
    assert(len(point1) == len(point2))
    sum = 0
    for p1, p2 in zip(point1, point2):
        sum += np.power((p1-p2),2)
    return np.power(sum,.5)

    


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    raise NotImplementedError


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        raise NotImplementedError


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        raise NotImplementedError