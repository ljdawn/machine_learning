"""
====================
unittest for tool box
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-02-17"""

__all__ = ['']
import unittest
import numpy as np
import Step_Zero_Processing_v06
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

class test_Zero_v06(unittest.TestCase):
	def setUp(self):
		"""a real data like data matrix, every element in the matrix must be int/float"""
		self.table = [[1.3,2,3,1,6],[0.0,5,7,2,7],[1.0,-100,20,3,7]]
		self.preprocess = Step_Zero_Processing_v06.preprocess_one
		print 'Step_Zero_Processing_v06 testing starting ... '
		print 'Function', self.preprocess, 'is loaded.'
		print 'Test data matirx :', self.table
	def test_preprocess_one(self):
		print 'Step_Zero_Processing_v06.preprocess_one() testing starting ... '
		print 'Test 1 --> stand model : 0, discret_list = [], binar_list = [], binar_thr_list = []'
		print 'Expecting : '
		print min_max_scaler.fit_transform(self.table)
		print 'Actually : '
		print self.preprocess(self.table)[0]
		self.assertEqual(min_max_scaler.fit_transform(self.table).tolist(), self.preprocess(self.table)[0].tolist())
		print 'Test 2 --> stand model : 0, discret_list = [5], binar_list = [], binar_thr_list = []'
		print 'Expecting : '
		print min_max_scaler.fit_transform(self.table)
		print 'Actually : '
		print self.preprocess(self.table)[0]
		self.assertEqual(min_max_scaler.fit_transform(self.table).tolist(), self.preprocess(self.table)[0].tolist())

if __name__ == '__main__':
	unittest.main()