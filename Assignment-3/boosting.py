import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		preds = []
		for xn in features:
			running_sum = 0.
			for clf, beta in zip(self.clfs_picked, self.betas):
				running_sum += beta * clf.predict(xn)
			if running_sum >= 0:
				preds.append(1)
			else:
				preds.append(-1)
		return preds
		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		N = len(features)
		w = [ 1. / N ] * N
		for t in range(0,self.T):
			min_classifier = None
			min_epsion = None
			# find best classifier
			for clf in self.clfs:
				epsilon = 0
				for wn, xn, yn in zip(w, features, labels):
					epsilon += wn * (1 if clf.predict(xn) != yn else 0)		
				if min_epsion == None:
					min_epsion = epsilon
				else:
					if min_epsion > epsilon: 
						min_epsion = epsilon
						min_classifier = clf

			self.clfs_picked.append(min_classifier)
			beta = 1 / 2 * np.log( (1 - min_epsion) / epsilon)
			self.betas.append(beta)

			w = [ wn * np.exp(-beta) if yn == min_classifier.predict(xn) else wn * np.exp(beta) for wn, xn, yn in zip(w, features, labels) ]
			w_sum = sum(w)
			w = [ wn / w_sum for wn  in w ]

		
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)


class LogitBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "LogitBoost"
		return

	def train(self, features: List[List[float]], labels: List[int]):
		N = len(features)
		pis = [1. / 2] * N
		for t in range(0,self.T):
			
			z = [ ((yn + 1) / 2 - pin) / (pin * (1 - pin)) for yn,pin in zip(labels,pis) ]
			w = [ pin * (1 - pin)  for pin in pis ]

			min_classifier = None
			min_value = None
			for clf in self.clfs:
				value = 0
				for wn, zn, xn in zip(w,z, features):
					value += wn * (zn - clf.predict(xn)) ** 2
				if min_value == None:
					min_value = value
					min_classifier = clf
				else:
					if min_value > value:
						min_value = value
						min_classifier = clf

			self.clfs_picked.append(min_classifier)
			self.betas.append(1 / 2)

			preds = self.predict(features)
			pis = [ 1 / (1 + np.exp(-2 * p)) for xn, p in zip(features, preds)]
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)
	