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
	def test_preprocess_one(self):
		print 'Step_Zero_Processing_v06.preprocess_one() test starting ... '
		print 'Test 1 --> stand model : 0, discret_list = [], binar_list = [], binar_thr_list = []'
		#print 'Expecting : '
		#print min_max_scaler.fit_transform(self.table)
		#print 'Actually : '
		#print self.preprocess(self.table)[0]
		self.assertEqual(min_max_scaler.fit_transform(self.table).tolist(), self.preprocess(self.table)[0].tolist())
		print 'Test 2 --> stand model : 0, discret_list = [3, 4], binar_list = [], binar_thr_list = []'
		Exp = [[1.0, 0.9714285714285715, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.23529411764705882, 0.0, 1.0, 0.0, 0.0, 1.0], [0.7692307692307692, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]]
		self.assertEqual(Exp, self.preprocess(self.table, 0, [3, 4])[0].tolist())
		print 'Test 3 --> stand model : 0, discret_list = [3, 4], binar_list = [1, 2], binar_thr_list = []'
		Exp = [[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], [0.7692307692307692, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]]
		self.assertEqual(Exp, self.preprocess(self.table, 0, [3, 4], [1, 2])[0].tolist())
		print 'Test 4 --> stand model : 0, discret_list = [3, 4], binar_list = [1, 2], binar_thr_list = [3, 19]'
		Exp = [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], [0.7692307692307692, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]]
		self.assertEqual(Exp, self.preprocess(self.table, 0, [3, 4], [1, 2], [3, 19])[0].tolist())
		print 'Test 5 --> stand model : 1, discret_list = [], binar_list = [], binar_thr_list = []'
		Exp = preprocessing.StandardScaler().fit(self.table).transform(self.table)
		self.assertAlmostEqual((Exp - self.preprocess(self.table, 1)[0]).sum(), 0)
		print 'Test 6 --> stand model : 1, discret_list = [], binar_list = [3, 4], binar_thr_list = [], test manually'
		print 'Test 7 --> stand model : 1, discret_list = [], binar_list = [3, 4], binar_thr_list = [1, 2], test manually'
		print 'Test 8 --> stand model : 2, discret_list = [], binar_list = [], binar_thr_list = []'
		Exp = preprocessing.StandardScaler().fit(self.table).transform(self.table)
		self.assertAlmostEqual((Exp - self.preprocess(self.table, 2)[0]).sum(), 0)
		print 'Test 9 --> stand model : 2, discret_list = [], binar_list = [3, 4], binar_thr_list = [], test manually'
		print 'Test 10 --> stand model : 2, discret_list = [], binar_list = [3, 4], binar_thr_list = [1, 2], test manually'

	def test_preprocess_one_exception(self):
		print '\nStep_Zero_Processing_v06.preprocess_one() Exception test starting ... '
		print 'Test 1 --> stand model : 0, discret_list = ["s"], binar_list = ["s"], binar_thr_list = ["s"]' 
		self.assertRaises(TypeError, self.preprocess, self.table, 0, ["s"])
		self.assertRaises(AssertionError, self.preprocess, self.table, 0, [None])
		self.assertRaises(TypeError, self.preprocess, self.table, 0, [], ["s"])
		self.assertRaises(AssertionError, self.preprocess, self.table, 0, [], [None])
		self.assertRaises(TypeError, self.preprocess, self.table, 0, [], [], ["s"])
		self.assertRaises(AssertionError, self.preprocess, self.table, 0, [], [], [None])
		print 'Test 2 --> stand model : 0, list over range'
		self.assertRaises(AssertionError, self.preprocess, self.table, 0, [5], [], [])
		self.assertRaises(AssertionError, self.preprocess, self.table, 0, [], [5], [])
		self.assertRaises(ValueError, self.preprocess, self.table, 0, [4], [4], [])
		self.assertRaises(AssertionError, self.preprocess, self.table, 0, [4], [4], [1, 2])
		self.assertRaises(AssertionError, self.preprocess, self.table, 0, [-1], [], [])
		self.assertRaises(AssertionError, self.preprocess, self.table, 0, [], [-1], [])


if __name__ == '__main__':
	unittest.main()