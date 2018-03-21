import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the dim of feature to be splitted

		self.feature_uniq_split = None # the feature to be splitted


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
			'''
			C = len(branches)
			B = len(branches[0])
			total = np.sum(branches)
			weighted_entropy = 0.

			for b in range(0,B):
				sum_branch = np.sum([branch[b] for branch in branches])
				running_entropy = 0.
				for c in range(0,C):
					total_c = branches[c][b]
					if total_c != 0:
						running_entropy += (total_c / sum_branch) * np.log((total_c / sum_branch))
				weighted_entropy += (sum_branch / total) * -running_entropy

			return weighted_entropy

		if len(self.features[0]) == 0:
			self.splittable = False
			return
		
		# #  compare each split using conditional entropy
		# #       find the best split
		cond_entropies = []

		for idx_dim in range(len(self.features[0])):
			feature_n = np.unique([feature[idx_dim] for feature in self.features])

			feature_to_index = {}
			for f, i in zip(feature_n, range(0,len(feature_n))):
				feature_to_index[f] = i

			branches = np.zeros((self.num_cls, len(feature_n)))
			labels_n = np.unique(self.labels).tolist()
			for f, l in zip(self.features, self.labels):
				l_i = labels_n.index(l)
				branches[l_i][feature_to_index[f[idx_dim]]] += 1

			c_entropy_n = conditional_entropy(branches)
			cond_entropies.append(c_entropy_n)
		max_entropy_feature_ind = np.argmax(cond_entropies)
		
		# split the node, add child nodes
		self.dim_split = max_entropy_feature_ind
		features_dim = [ feature[max_entropy_feature_ind] for feature in self.features]
		feature_dim_to_split = np.unique(features_dim)
		self.feature_uniq_split = feature_dim_to_split.tolist()
		for f_s in feature_dim_to_split:
			node_features = []
			node_labels = []

			for f, l in zip(self.features, self.labels):
				if f[max_entropy_feature_ind] == f_s:
					node_features.append(f)
					node_labels.append(l)

			node_features = np.delete(node_features, max_entropy_feature_ind, 1).tolist()
			node = TreeNode(node_features, node_labels, len(np.unique(node_labels)))
			self.children.append(node)

		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			# print(feature)
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



