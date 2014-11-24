#cording:utf-8

import pandas as pd
import numpy as np
import collections as co 

class Mydata(object):

	def __init__(self, data_path, columns_path, sep = '\t'):
		self.columns = []
		self.re_dict = {}
		self.data_path = data_path
		self.columns_path = columns_path
		self.sep = sep

	def load_file(self):
		for line in open(self.columns_path):
			key, value = line.strip().split(self.sep)
			self.columns.append((key, value))
			if value not in self.re_dict:
				self.re_dict[value] = []
				self.re_dict[value].append(key)
			else:
				self.re_dict[value].append(key)
		self.columns_dict = co.OrderedDict(self.columns)
		self.data = pd.read_csv(self.data_path, names = self.columns_dict.keys(), sep = self.sep)
		self.data = self.data[self.data['Y'] < 2]
		return (self.data, self.re_dict)

if __name__ == '__main__':
	pass