"""
====================
data processing 
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-01-17"""

__all__ = ['']

#coding:utf8
from operator import *

def get_box(value_list, count = 10):
	import numpy as np
	import scipy as sp
	value_list.sort(reverse = True)
	n_con = np.array(value_list)
	cou, res = sp.histogram(n_con,count)
	return (cou, res, value_list, map(lambda x:x[1],filter(lambda (x, y):x%(len(n_con)/count) == 0, enumerate(n_con)))[1::])

def my_Discretization(*args):
	assert len(args[2]) - len(args[1]) in (1, 0)
	assert len(args) == 4
	key = args[0]
	def inner(key_s):
		flg = -1
		t_res = [0] * len(args[2])
		for i in xrange(len(args[1])):
			if args[-1](key_s, args[1][i]):
				flg = i
				break
		t_res[flg] = 1
		return t_res
	return map(inner, key)

def my_Standardization(*args):
	from sklearn import preprocessing
	import numpy as np
	assert len(args) in (1,2)
	return preprocessing.scale(np.array(args[0])) if len(args) == 1 else preprocessing.StandardScaler().fit(np.array(args[0])).transform(np.array(args[1])) 

def my_binarizer(*args):
	from sklearn import preprocessing
	import numpy as np
	assert len(args) == 2
	threshold = args[1]
	binarizer = preprocessing.Binarizer(threshold)
	return binarizer.transform(np.array(args[0]))

def preprocess(data_matrix, stand_flag = 0, discret_list = [], binar_list = [], binar_thr_list = []):
	"""data_matrix : two-dimensional array; feature matrix.
		stand_flag : standardization flag; 0->non-standardization-->range(0,1); 1->standardize full matrix in last stage; 2 ->standardize no-change columns; 
					*0 -> range(0, 1), 0/1;range(0,1), 0/1
					*1 -> mean --> 0 var --> 1
					*2 -> mean --> 0 var --> 1, 0/1;range(0,1),0/1
		discret_list : the column to discretize; list of column_num; like [1,2,3,4] 
		binar_list : the column to binarize; list of column_num; like [1,2,3,4]
		binar_thr_list : threshold to be binarized *defult -> mean of column 
	"""
	import numpy as np
	from sklearn import preprocessing
	enc = preprocessing.OneHotEncoder()
	min_max_scaler = preprocessing.MinMaxScaler()
	#init
	target = np.array(data_matrix)
	(m, n) = target.shape
	box = []
	target_binarizer_final_count = 0
	categorize_class = np.array([])
	target_filter = np.zeros(n)
	#class 1 : categorize, 2 : binarize
	for num in discret_list:
		target_filter[num] = 1
	for num in binar_list:
		target_filter[num] = 2
	if len(discret_list) != 0:
		target_discret = target.T[target_filter == 1].T
		categorize_class = map(lambda x:len(set(x)), target_discret.T)
		categorize_sizes = sum(categorize_class)
		discret_adjust = target_discret.min()
		enc.fit(target_discret - discret_adjust)
		target_discret_final = enc.transform(target_discret - discret_adjust).toarray()
		box.append(target_discret_final)
	if len(binar_list) != 0:
		target_binarizer = target.T[target_filter == 2]
		if binar_thr_list == []:
			binar_thr_list = [x.mean() for x in target_binarizer]
		target_binarizer_final = np.vstack(map(lambda l:preprocessing.Binarizer(threshold = binar_thr_list[l]).transform(target_binarizer[l]), range(len(binar_list)))).T
		target_binarizer_final_count = target_binarizer_final.shape[1]
		box.append(target_binarizer_final)
	if stand_flag != 2:
		target_other = min_max_scaler.fit_transform(target.T[target_filter == 0].T)
		box.append(target_other)
		if len(box) != 1:
			target_step_1 = np.hstack(box[::-1]) 
		else:
			target_step_1 = target_other
		if stand_flag == 0:
			res = target_step_1 
		elif stand_flag == 1: 
			res = preprocessing.scale(target_step_1)
	else:
		target_other = preprocessing.scale(target.T[target_filter == 0].T) 
		box.append(min_max_scaler.fit_transform(target_other))
		if len(box) != 1:
			res = np.hstack(box[::-1]) 
		else:
			res = target_other
	preprocessing_summary = {'categorize_class':categorize_class, \
	'no change column':(0, target_other.shape[1]), \
		'binarized column':(target_other.shape[1], target_other.shape[1]+target_binarizer_final_count), \
		'discret column':(n-len(categorize_class),n-len(categorize_class)+sum(categorize_class)), \
		'standardization':stand_flag}
	return (res, preprocessing_summary)

def column_picker(data_matrix, column_to_pick = []):
	import numpy as np
	target = np.array(data_matrix)
	assert max(column_to_pick) < target.shape[1]
	target_step_1 = filter(lambda (x,y):x in column_to_pick, enumerate(target.T))
	return np.vstack([x[1].T for x in target_step_1]).T

def column_rearrange_num(data_matrix, new_order):
	import numpy as np
	new_order_dict = dict([(y, x) for (x, y) in enumerate(new_order)]) 
	ori_list_dict = dict(enumerate(np.array(data_matrix).T))
	new_matrix_list = [(new_order_dict[i], ori_list_dict[i]) for i in xrange(len(new_order))]
	new_matrix_list.sort()
	return np.array([y for (x, y) in new_matrix_list]).T

def column_get_label_num(ori_label, new_label):
	ori_label_dict = dict([(y, x) for (x, y) in enumerate(ori_label)])
	return [ori_label_dict[x] for x in new_label]

if __name__ == '__main__':
	#1---
	#box = [5, 4, 3, 2, 1]
	#value = ['a','b','c','d','e']
	##ge --> '>=''
	#op = ge
	#to_prepare = [6, 5, 4, 3, 2, 1, 0, 5, 6, 7, 10, 10, 10, 10, 10, 10, 10]
	#print get_box(to_prepare, 5)
	#box = get_box(to_prepare, len(value))[3]
	#print my_Discretization(to_prepare, box, value, op)
	
	#2--
	#train = [[ 1., -1.,  2.], [ 2.,  0.,  0.], [ 0.,  1., -1.]]
	#to_prepare = [[-1.,  1., 0.]]
	#print my_Standardization(train)
	#print my_Standardization(train, to_prepare)
	
	#3--
	#t = 2
	#to_prepare = [0,1,2,3,4,5]
	#print my_binarizer(to_prepare, t)

	#4--
	table = [[1.3,2,3,1,6],[0.0,5,7,2,7],[1.0,-100,20,3,7]]
	for item in table:
		print item
	#print preprocess(table, discret_list = [3, 4], binar_list = [1, 2], binar_thr_list = [0, 10])
	#print preprocess(table, discret_list = [3, 4], binar_list = [1, 2])
	print preprocess(table, stand_flag = 2, discret_list = [3, 4], binar_list = [1, 2])
	#print preprocess(table, discret_list = [3, 4])
	#print preprocess(table, binar_list = [1, 2], binar_thr_list = [0, 10])
	#print preprocess(table, stand_flag = 1)
	#print preprocess(table, stand_flag = 0, discret_list = [3, 4], binar_list = [1, 2], binar_thr_list = [0, 10])
	#print preprocess(table)
	#5-- column_picker
	#print column_picker(table, [1,2,3,4])
	#6-- column rearrange
	#print column_rearrange_num(table, [1, 4, 3, 0, 2])
	#7-- column rearrange label
	#ori = ['a', 'b', 'c', 'd', 'e']
	#new = ['b', 'e', 'd', 'a', 'c']
	#print column_get_label_num(ori, new)

