#cording:utf-8

import numpy as np
import cPickle as pickle
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class Trainer(object):

	def __init__(self, dataset, model_path = 'model/model.pickle'):
		self.y = np.array(dataset.iloc[:,0])
		self.X = np.array(dataset.iloc[:,1:])
		self.modeler = linear_model.LogisticRegression()
		self.model_path = model_path

	def train(self):
		self.model = self.modeler.fit(self.X, self.y)

	def save_model(self):
		with open(self.model_path,'wb') as my_model:
			pickle.dump(self.model, my_model)

	def show_res(self):
		self.y_ = self.model.predict(self.X)
		self.Y_pre_prob = [b for [a, b] in self.model.predict_proba(self.X)]
		print (self.y_, self.Y_pre_prob)

	def _report(self):
		return(confusion_matrix(self.y, self.y_),classification_report(self.y, self.y_, target_names = ['class 1', 'class 2']))

	def show_report(self):
		print 'confusion_matrix:'
		print 'CMAT', '[0]','----', '[1]'
		print '[0]', '|', self._report()[0][0][0], '\t', self._report()[0][0][1], '|'
		print '[1]', '|', self._report()[0][1][0], '\t', self._report()[0][1][1], '|'

