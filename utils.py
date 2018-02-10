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
    # precsion = true positives / positives pred
    # recall = true positives classified / total positives 
    # 2 * [ (precision * recall) / (precision + recall) ]
    assert len(real_labels) == len(predicted_labels)
    truePositPred = sum(1 if (p == 1 and r == p) else 0 for r,p in zip(real_labels, predicted_labels))
    positPred = sum(1 if p ==1 else 0 for p in predicted_labels)
    precision = truePositPred / positPred

    totalPosit = sum(1 if r == 1 else 0 for r in real_labels)
    recall = truePositPred / totalPosit
    if (precision + recall) == 0:
        return 0
    return 2 * ((precision * recall) / (precision+recall))
    


def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:
    raise NotImplementedError


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    assert(len(point1) == len(point2))
    # d(x,y)=√⟨x−y)*(x−y⟩
    x = np.array(point1)
    y = np.array(point2)

    return np.power(np.subtract(x,y).dot(np.subtract(x,y)),.5)


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    assert len(point1) == len(point2)
    # d(x,y)=⟨x,y⟩
    x = np.array(point1)
    y = np.array(point2)
    return x.dot(y)

def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    # d(x,y)=exp(−1/2√⟨x−y,x−y⟩)
    assert len(point1) == len(point2)
    x = np.array(point1)
    y = np.array(point2)
    return np.exp(-.5 * (np.power(np.subtract(x,y).dot(np.subtract(x,y)),.5)))



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