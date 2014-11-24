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

	def look_up(self):
		return self.data


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

	def look_up(self):
		return self.data

class Colcombine(object):

	def __init__(self, noun_block, num_block):
		self.values = np.hstack((noun_block.values, num_block.values))
		self.labels = noun_block.columns + num_block.columns

	def look_up(self):
		self.data = pd.DataFrame(self.values, columns = self.labels)
		print self.data



a = np.array([[1,2,3,4,66],[2,3,4,5,-1],[3,4,5,100,89]])
title = ['a', 'b', 'c', 'd', 'e']
c = pd.DataFrame(a, columns = title)
print c

noun_list = ['a', 'b', 'c', 'd']
num_list = ['e']

df_noun = c[noun_list]
df_num = c[num_list]

dfnoun = Colnoun(df_noun)
dfnum = Colnum(df_num)
dfnoun.one_hot()
b = dfnoun.look_up()
dfnum.imp()
dfnum.scale()
a = dfnum.look_up()

print b, a

d = Colcombine(b, a)
d.look_up()
