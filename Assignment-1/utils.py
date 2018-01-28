from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)

    raise NotImplementedError


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
    raise NotImplementedError


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    raise NotImplementedError


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    raise NotImplementedError


def normalize(features: List[List[float]]) -> List[List[float]]:
    """
    normalize the feature vector for each sample . For example,
    if the input features = [[3, 4], [1, -1], [0, 0]],
    the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
    """
    raise NotImplementedError


def min_max_scale(features: List[List[float]]) -> List[List[float]]:
    """
    normalize the feature vector for each sample . For example,
    if the input features = [[2, -1], [-1, 5], [0, 0]],
    the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
    """
    raise NotImplementedError
