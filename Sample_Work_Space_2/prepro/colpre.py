#cording:utf-8

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn import linear_model

class Colnoun(object):

	def __init__(self, pd_blocks):
		self.data = pd_blocks

	def _one_hot_data(self):
		enc = OneHotEncoder()
		enc.fit(self.data.values)
		return enc.transform(self.data.values).toarray()

	def _one_hot_labels(self):
		one_hot_labels_l = []
		for key in self.data.columns:
			one_hot_labels_l.extend([(key + '_' + str(label)) for label in self.data[key].unique()])
		return one_hot_labels_l

	def one_hot(self):
		self.data = pd.DataFrame(data = self._one_hot_data(), columns = self._one_hot_labels())

	def get_data(self):
		return self.data

	def look_up(self):
		pass


class Colnum(object):

	def __init__(self, pd_blocks):
		self.data = pd_blocks

	def imp(self, missing_values=-1, strategy='mean', axis=0):
		imp = Imputer(missing_values, strategy, axis)
		imp.fit(self.data.values)
		self.data = pd.DataFrame(imp.transform(self.data.values), columns = self.data.columns)

	def scale(self):
		min_max_scaler = preprocessing.MinMaxScaler()
		scaler = preprocessing.StandardScaler().fit(self.data.values)
		self.data = pd.DataFrame(min_max_scaler.fit_transform(scaler.transform(self.data.values)), columns = self.data.columns)

	def get_data(self):
		return self.data

class Colcombine(object):

	def __init__(self, Y, noun_block, num_block):
		self.values = np.hstack((Y.values, noun_block.values, num_block.values))
		self.labels = Y.columns + noun_block.columns + num_block.columns

	def get_data(self):
		self.data = pd.DataFrame(self.values, columns = self.labels)
		return self.data

if __name__ == '__main__':
	pass