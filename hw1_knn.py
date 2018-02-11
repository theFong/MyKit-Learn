from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        assert len(features) == len(labels)
        self.feature_and_label = list(zip(features,labels))

    def predict(self, features: List[List[float]]) -> List[int]:
        # calculate nn for all many points

        return [ self.nn(f, self.k) for f in features ]
        	
    def nn(self, new_f: List[float], k: int) -> int:
        assert k > 0
        # calculate nn for point
        # calc distances sort take quorum based off K
        distances = [self.distance(f,new_f) for f in self.feature_and_label]
        distances = sorted(distances, key = lambda x: x[0])
        
        label_bucket = {0:0,1:0}
        for i in range(0,k):
            if i >= len(distances):
                break
            vote = distances[i][1]
            if vote not in label_bucket:
                label_bucket[vote] = 1
            else:
                label_bucket[vote] += 1

        winner = max(label_bucket, key = label_bucket.get)
        winner_count = label_bucket[winner]
        # check to see if there is a tie
        checkTie = sum(1 if v == winner_count else 0 for k,v in label_bucket.items())
        # if tie call nn again with k-1
        if checkTie > 1:
            return self.nn(new_f,k-1)
        else:
            return winner

    def distance(self, point: (List[float],int), new_p: List[float]):
    	return (self.distance_function(point[0],new_p), point[1])


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
