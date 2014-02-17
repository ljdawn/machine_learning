"""
====================
data processing 
working with numpy, pandas, scikit-learn
*stand -1 test can not be use
*test data(after data preprocessing) columns must = training data(after datap rocessing)
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-02-14"""

__all__ = ['']

#coding:utf8
from operator import *

def preprocess_one(data_matrix, stand_flag = 0, discret_list = [], binar_list = [], binar_thr_list =[]):
	"""discret_list, binar_list, binar_thr_list, must be list of init/float"""
	import numpy as np
	from sklearn import preprocessing
	matrix_catalog_converter = preprocessing.OneHotEncoder()
	min_max_scaler = preprocessing.MinMaxScaler()#[0, 1] by default
	#init
	to_process = np.array(data_matrix)
	(m, n) = to_process.shape#m -> rows, n -> cols
	process_parts_filter = np.zeros(n)#clos init by 0, class-> 0 : continuous, 1 : categorize, 2 : binarize
	if discret_list != []:
		process_parts_filter[discret_list[0] : discret_list[0] + len(discret_list)] = [2] * len(discret_list)
	if binar_list != []:
		process_parts_filter[binar_list[0] : binar_list[0] + len(binar_list)] = [1] * len(binar_list)
	total_box = []
	categorize_class = np.array([])
	target_binarizer_final_count = 0
	#processing
	if discret_list != []:
		target_discret = to_process.T[process_parts_filter == 2].T
		categorize_class = map(lambda x:len(set(x)), target_discret.T)
		categorize_sizes = sum(categorize_class)
		discret_adjust = target_discret.min()
		matrix_catalog_converter.fit(target_discret - discret_adjust)
		target_discret_final = matrix_catalog_converter.transform(target_discret - discret_adjust).toarray()
		total_box.append(target_discret_final)
	if binar_list != []:
		target_binarizer = to_process.T[process_parts_filter == 1]
		if binar_thr_list == []:
			binar_thr_list = [x.mean() for x in target_binarizer]
		target_binarizer_final = np.vstack(map(lambda l:preprocessing.Binarizer(threshold = binar_thr_list[l]).transform(target_binarizer[l]), range(len(binar_list)))).T
		target_binarizer_final_count = target_binarizer_final.shape[1]
		total_box.append(target_binarizer_final)

	if stand_flag != 2:
		target_other = min_max_scaler.fit_transform(to_process.T[process_parts_filter == 0].T)
		total_box.append(target_other)
		if len(total_box) != 1:
			target_union = np.hstack(total_box[::-1]) 
		else:
			target_union = target_other
		if stand_flag == 0:
			res = target_union
		elif stand_flag == 1:
			res = preprocessing.StandardScaler().fit(target_union).transform(target_union)
	else:
		target_other = preprocessing.scale(to_process.T[process_parts_filter == 0].T)
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

def preprocess(data_matrix, data_matrix_test = '', stand_flag = 0, test_flag = False, discret_list = [], binar_list = [], binar_thr_list = []):
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
	#encoding categorical features -> binarization
	min_max_scaler = preprocessing.MinMaxScaler()
	#scaling -> [0, 1]
	target = np.array(data_matrix)
	(m, n) = target.shape
	#m -> col, n -> row
	box = []
	target_binarizer_final_count = 0
	categorize_class = np.array([])
	target_filter = np.zeros(n)
	t_l = len(data_matrix) - len(data_matrix_test)
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
		if test_flag == False:
			box.append(target_discret_final)
		else:
			box.append(target_discret_final[t_l:])	
	if len(binar_list) != 0:
		target_binarizer = target.T[target_filter == 2]
		if binar_thr_list == []:
			binar_thr_list = [x.mean() for x in target_binarizer]
		target_binarizer_final = np.vstack(map(lambda l:preprocessing.Binarizer(threshold = binar_thr_list[l]).transform(target_binarizer[l]), range(len(binar_list)))).T
		target_binarizer_final_count = target_binarizer_final.shape[1]
		if test_flag == False:
			box.append(target_binarizer_final)
		else:
			box.append(target_binarizer_final[t_l:])
	if stand_flag != 2:
		if test_flag == False:
			target_other = min_max_scaler.fit_transform(target.T[target_filter == 0].T)
		else:
			target_other = min_max_scaler.fit_transform(target.T[target_filter == 0].T[0:t_l])
			target_other = min_max_scaler.transform(target.T[target_filter == 0].T[t_l:])
		box.append(target_other)
		if len(box) != 1:
			target_step_1 = np.hstack(box[::-1]) 
		else:
			target_step_1 = target_other
		if stand_flag == 0:
			res = target_step_1 
		elif stand_flag == 1:
			if test_flag == False:
				res = preprocessing.StandardScaler().fit(target_step_1).transform(target_step_1) 
			else:
				scaler = preprocessing.StandardScaler().fit(target_step_1[0:t_l])
				res = scaler.transform(target_step_1[t_l:])
	else:
		if test_flag == False:
			target_other = preprocessing.scale(target.T[target_filter == 0].T)
		else:
			target_other = preprocessing.StandardScaler().fit(target.T[target_filter == 0].T[0:t_l]).transform(target.T[target_filter == 0].T[t_l:])
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
	"""to pick columns by ids"""
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

def column_picker_pandas(data_matrix, column_to_pick = []):
	"""to pick columns by column name"""
	return data_matrix[column_to_pick]

column_rearrange_num_pandas = column_picker_pandas

if __name__ == '__main__':
	import numpy as np
	import pandas as pd
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
	table = [[1/2,2,3,1,-6],[0.0,5,7,2,7],[1.0,-100,20,3,7]]
	#--table_test = [[3,3,3,3,3], [4,4,4,4,4]]
	#for item in table:
	#	print item
	#--table_ =  np.array(table)
	#--table_test_ = np.array(table_test)
	#--data = {'a':table_[:,0], 'b':table_[:,1], 'c':table_[:,2], 'd':table_[:,3], 'e':table_[:,4]}
	#--data_test = {'a':table_test_[:,0], 'b':table_test_[:,1], 'c':table_test_[:,2], 'd':table_test_[:,3], 'e':table_test_[:,4]}
	#--table_frame = pd.DataFrame(data)
	#--table_frame_test = pd.DataFrame(data_test)
	#print preprocess(table, discret_list = [3, 4], binar_list = [1, 2], binar_thr_list = [0, 10])
	#print preprocess(table, discret_list = [3, 4], binar_list = [1, 2])
	#print preprocess(table, stand_flag = 2, discret_list = [3, 4], binar_list = [1, 2])
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
	#8 pandas
	#print column_picker_pandas(table_frame, ['a','c'])
	#print column_picker(table, column_get_label_num(['a', 'b', 'c', 'd', 'e'], ['a', 'c']))
	#9 preprocess for test
	#--print table_frame
	#--print table_frame_test
	#--print preprocess(table_frame, stand_flag = 0, discret_list = [3, 4])[0].tolist()
	#print preprocess(table_frame_test, stand_flag = 0, discret_list = [3, 4],  binar_thr_list = [0, 10])[0].tolist()
	#--data_matrix = pd.concat([table_frame, table_frame_test])
	#--print preprocess(data_matrix, table_frame_test, test_flag = True, stand_flag = 0, discret_list = [3, 4])[0].tolist()
	for item in table:
		print item
	print preprocess_one(table, discret_list = [3, 4], binar_list = [1, 2])[0]
	print preprocess_one(table)[0]
