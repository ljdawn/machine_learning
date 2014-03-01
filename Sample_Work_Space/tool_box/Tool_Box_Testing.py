"""
====================
unittest for tool box
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-02-17"""

__all__ = ['']
import unittest
import logging
import numpy as np
import Step_Zero_Processing_v07
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

class test_Zero_v07(unittest.TestCase):
	def setUp(self):
		"""a real data like data matrix, every element in the matrix must be int/float"""
		self.table = [[1.3,2,3,1,6],[0.0,5,7,2,7],[1.0,-100,20,3,7]]
		self.table_tr = [[1.3,2,3,1,6],[0.0,5,7,2,7]]
		self.table_te = [[1.0,-100,20,3,7]]
		self.table_1 = [[10, 9]]
		self.table_tr_1 = [[1.3,2,3,1,6],[0.0,5,7,2,7]]
		self.table_te_1 = [[1.0,-100,20,3,7]]
		self.preprocess_ori = Step_Zero_Processing_v07.preprocess
		self.preprocess = Step_Zero_Processing_v07.preprocess_one
		self.column_picker = Step_Zero_Processing_v07.column_picker
		self.column_rearrange_num = Step_Zero_Processing_v07.column_rearrange_num
		self.column_get_label_num = Step_Zero_Processing_v07.column_get_label_num
	def test_preprocess(self):
		logging.info('Step_Zero_Processing_v07.preprocess() test ...')
		logging.debug('Test 1 --> stand model : 0, discret_list = [], binar_list = [], binar_thr_list = []')
		Exp = self.preprocess(self.table)[0][:(len(self.table)-len(self.table_te))]
		funcR = self.preprocess_ori(self.table, self.table_te, test_flag = False)[0]
		self.assertAlmostEqual((np.array(Exp)-funcR).sum(), 0)
		Exp = self.preprocess(self.table)[0][(len(self.table)-len(self.table_te)):]
		funcR = self.preprocess_ori(self.table, self.table_te, test_flag = True)[0]
		self.assertAlmostEqual((np.array(Exp)-funcR).sum(), 0)
		logging.debug('Test 2 --> stand model : 0, discret_list = [3, 4], binar_list = [], binar_thr_list = []')
		Exp = self.preprocess(self.table, 0, [3, 4])[0][:(len(self.table)-len(self.table_te))]
		funcR = self.preprocess_ori(self.table, self.table_te, test_flag = False, discret_list = [3, 4])[0]
		self.assertAlmostEqual((np.array(Exp)-funcR).sum(), 0)
		Exp = self.preprocess(self.table, 0, [3, 4])[0][(len(self.table)-len(self.table_te)):]
		funcR = self.preprocess_ori(self.table, self.table_te, test_flag = True, discret_list = [3, 4])[0]
		self.assertAlmostEqual((np.array(Exp)-funcR).sum(), 0)
	def test_preprocess_exception(self):
		logging.info('Step_Zero_Processing_v07.preprocess_exception() test ...')
	def test_preprocess_one(self):
		logging.info('Step_Zero_Processing_v07.preprocess_one() test ...')
		logging.debug('Test 1 --> stand model : 0, discret_list = [], binar_list = [], binar_thr_list = []')
		#print 'Expecting : '
		#print min_max_scaler.fit_transform(self.table)
		#print 'Actually : '
		#print self.preprocess(self.table)[0]
		self.assertEqual(min_max_scaler.fit_transform(self.table).tolist(), self.preprocess(self.table)[0].tolist())
		logging.debug('Test 2 --> stand model : 0, discret_list = [3, 4], binar_list = [], binar_thr_list = []')
		Exp = [[1.0, 0.9714285714285715, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.23529411764705882, 0.0, 1.0, 0.0, 0.0, 1.0], [0.7692307692307692, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]]
		self.assertEqual(Exp, self.preprocess(self.table, 0, [3, 4])[0].tolist())
		logging.debug('Test 3 --> stand model : 0, discret_list = [3, 4], binar_list = [1, 2], binar_thr_list = []')
		Exp = [[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], [0.7692307692307692, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]]
		self.assertEqual(Exp, self.preprocess(self.table, 0, [3, 4], [1, 2])[0].tolist())
		logging.debug('Test 4 --> stand model : 0, discret_list = [3, 4], binar_list = [1, 2], binar_thr_list = [3, 19]')
		Exp = [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], [0.7692307692307692, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]]
		self.assertEqual(Exp, self.preprocess(self.table, 0, [3, 4], [1, 2], [3, 19])[0].tolist())
		logging.debug('Test 5 --> stand model : 1, discret_list = [], binar_list = [], binar_thr_list = []')
		Exp = preprocessing.StandardScaler().fit(self.table).transform(self.table)
		self.assertAlmostEqual((Exp - self.preprocess(self.table, 1)[0]).sum(), 0)
		logging.debug('Test 6 --> stand model : 1, discret_list = [], binar_list = [3, 4], binar_thr_list = [], test manually')
		logging.debug('Test 7 --> stand model : 1, discret_list = [], binar_list = [3, 4], binar_thr_list = [1, 2], test manually')
		logging.debug('Test 8 --> stand model : 2, discret_list = [], binar_list = [], binar_thr_list = []')
		Exp = preprocessing.StandardScaler().fit(self.table).transform(self.table)
		self.assertAlmostEqual((Exp - self.preprocess(self.table, 2)[0]).sum(), 0)
		logging.debug('Test 9 --> stand model : 2, discret_list = [], binar_list = [3, 4], binar_thr_list = [], test manually')
		logging.debug('Test 10 --> stand model : 2, discret_list = [], binar_list = [3, 4], binar_thr_list = [1, 2], test manually')
	def test_preprocess_one_exception(self):
		logging.info('Step_Zero_Processing_v07.preprocess_one() Exception test ... ')
		logging.debug('Test 1 --> stand model : 0, discret_list = ["s"], binar_list = ["s"], binar_thr_list = ["s"]') 
		self.assertRaises(TypeError, self.preprocess, self.table, 0, ["s"])
		self.assertRaises(AssertionError, self.preprocess, self.table, 0, [None])
		self.assertRaises(TypeError, self.preprocess, self.table, 0, [], ["s"])
		self.assertRaises(AssertionError, self.preprocess, self.table, 0, [], [None])
		self.assertRaises(TypeError, self.preprocess, self.table, 0, [], [], ["s"])
		self.assertRaises(AssertionError, self.preprocess, self.table, 0, [], [], [None])
		logging.debug('Test 2 --> stand model : 0, list over range')
		self.assertRaises(AssertionError, self.preprocess, self.table, 0, [5], [], [])
		self.assertRaises(AssertionError, self.preprocess, self.table, 0, [], [5], [])
		self.assertRaises(ValueError, self.preprocess, self.table, 0, [4], [4], [])
		self.assertRaises(AssertionError, self.preprocess, self.table, 0, [4], [4], [1, 2])
		self.assertRaises(AssertionError, self.preprocess, self.table, 0, [-1], [], [])
		self.assertRaises(AssertionError, self.preprocess, self.table, 0, [], [-1], [])
	def test_column_picker(self):
		logging.info('Step_Zero_Processing_v07.column_picker() test ... ')
		logging.debug('Test 1 --> self.table, column_to_pick = [1, 2]')
		Exp = [[2, 3],[5, 7],[-100, 20]]
		funcR = self.column_picker(self.table, [1, 2])
		self.assertAlmostEqual((np.array(Exp)-funcR).sum(), 0)
		logging.debug('Test 2 --> self.table, column_to_pick = [1, 3]')
		Exp = [[2, 1],[5, 2],[-100, 3]]
		funcR = self.column_picker(self.table, [1, 3])
		self.assertAlmostEqual((np.array(Exp)-funcR).sum(), 0)
		logging.debug('Test 3 --> self.table, column_to_pick = [0]')
		Exp = [[1.3],[0.0],[1.0]]
		funcR = self.column_picker(self.table, [0])
		self.assertAlmostEqual((np.array(Exp)-funcR).sum(), 0)
		logging.debug('Test 4 --> self.table, column_to_pick = [-1]')
		Exp = [[6],[7],[7]]
		funcR = self.column_picker(self.table, [4])
		self.assertAlmostEqual((np.array(Exp)-funcR).sum(), 0)
	def test_column_picker_exception(self):
		logging.info('Step_Zero_Processing_v07.test_column_picker_exception() test ... ')
		logging.debug('Test 1 --> self.table, column_to_pick = [7]')
		self.assertRaises(AssertionError, self.column_picker, self.table, [7])
		logging.debug('Test 2 --> self.table, column_to_pick = [-7]')
		self.assertRaises(AssertionError, self.column_picker, self.table, [-7])
		logging.debug('Test 3 --> self.table, column_to_pick = []/None/string')
		self.assertRaises(ValueError, self.column_picker, self.table, [])
		self.assertRaises(AssertionError, self.column_picker, self.table, [None])
		self.assertRaises(AssertionError, self.column_picker, self.table, ["s"])
	def test_column_rearrange_num(self):
		logging.info('Step_Zero_Processing_v07.test_column_rearrange_num() test ... ')
		logging.debug('Test 1 --> self.table, new_order = [4, 3, 2, 1, 0]')
		Exp = [line[::-1] for line in self.table]
		funcR = self.column_rearrange_num(self.table, [4, 3, 2, 1, 0])
		self.assertAlmostEqual((np.array(Exp)-funcR).sum(), 0)
		logging.debug('Test 2 --> self.table, new_order = [3, 4, 0, 1, 2]')
		Exp = [[line[3], line[4], line[0], line[1], line[2]] for line in self.table]
		funcR = self.column_rearrange_num(self.table, [4, 3, 2, 1, 0])
		self.assertAlmostEqual((np.array(Exp)-funcR).sum(), 0)
	def test_column_rearrange_num_exception(self):
		logging.info('Step_Zero_Processing_v07.test_column_rearrange_num_exception() test ... ')
		logging.debug('Test 1 --> self.table, new_order = [1, 0]/[1, 1, 2, 3, 4]/[0, 2, 1, 3, 5]/["s"]/[None]/[-1]')
		self.assertRaises(AssertionError, self.column_rearrange_num, self.table, [1, 0])
		self.assertRaises(AssertionError, self.column_rearrange_num, self.table, [1, 1, 2, 3, 4])
		self.assertRaises(AssertionError, self.column_rearrange_num, self.table, [0, 2, 1, 3, 5])
		self.assertRaises(AssertionError, self.column_rearrange_num, self.table, ['s'])
		self.assertRaises(AssertionError, self.column_rearrange_num, self.table, [None])
		self.assertRaises(AssertionError, self.column_rearrange_num, self.table, [-1])
	def test_column_get_label_num(self):
		logging.info('Step_Zero_Processing_v07.column_get_label_num() test ... ')
		logging.debug("Test 1 --> ori_label = ['a', 'b', 'c', 'd', 'e'], new_label = ['d', 'a', 'b', 'e', 'c']")
		Exp = [3, 0, 1, 4, 2]
		funcR = self.column_get_label_num(ori_label = ['a', 'b', 'c', 'd', 'e'], new_label = ['d', 'a', 'b', 'e', 'c'])
		self.assertEqual(Exp, funcR)

if __name__ == '__main__':
	logging.basicConfig(level = logging.INFO)
	unittest.main()

	