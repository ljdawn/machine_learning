"""
====================
data processing 
working with numpy, pandas, scikit-learn
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ = """2013-02-18"""

__all__ = ['']

#coding:utf8
from operator import *
import cProfile

def preprocess_one(data_matrix, stand_flag = 0, discret_list = [], binar_list = [], binar_thr_list =[]):
	"""data_matrix : two-dimensional array; feature matrix.
		stand_flag : standardization flag; 0->non-standardization-->range(0,1); 1->standardize full matrix in last stage; 2 ->standardize no-change columns; 
					*0 -> range(0, 1), 0/1|range(0, 1), 0/1-class|rang(0, 1)
					*1 -> mean --> 0 var --> 1
					*2 -> mean --> 0 var --> 1, 0/1|range(0,1), 0/1-class|range(0, 1)
		discret_list : the column to discretize; list of column_num; like [1,2,3,4] 
		binar_list : the column to binarize; list of column_num; like [1,2,3,4]
		binar_thr_list : threshold to be binarized *defult -> mean of column 
	"""
	import numpy as np
	#<<Test>>Input exception test
	assert np.array(data_matrix).sum()
	to_process = np.array(data_matrix)
	(m, n) = to_process.shape#m -> rows, n -> cols
	assert stand_flag in (0, 1, 2)
	if discret_list != []:
		ndl = np.array(discret_list)
		assert ndl.sum()
		assert ndl.max() < n
		assert ndl.min() >= 0
	if binar_list != []:
		nbl = np.array(binar_list)
		assert nbl.sum()
		assert nbl.max() < n
		assert nbl.min() >= 0
	if discret_list != [] and binar_list != []:
		assert discret_list not in binar_list
	if binar_thr_list != []:
		assert np.array(binar_thr_list).sum()
		assert len(binar_thr_list) == len(binar_list)
	from sklearn import preprocessing
	matrix_catalog_converter = preprocessing.OneHotEncoder()
	min_max_scaler = preprocessing.MinMaxScaler()#[0, 1] by default
	#init
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
		#total_box.append(min_max_scaler.fit_transform(target_other))
		total_box.append(target_other)
		if len(total_box) != 1:
			res = np.hstack(total_box[::-1]) 
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
		test_flag : False -> output:training data(upper), True - > output:test data(lower)  
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
			box.append(target_discret_final[:t_l])#training data part 3
		else:
			box.append(target_discret_final[t_l:])#test data part 3	
	if len(binar_list) != 0:
		target_binarizer = target.T[target_filter == 2]
		if binar_thr_list == []:
			binar_thr_list = [x.mean() for x in target_binarizer]
		target_binarizer_final = np.vstack(map(lambda l:preprocessing.Binarizer(threshold = binar_thr_list[l]).transform(target_binarizer[l]), range(len(binar_list)))).T
		target_binarizer_final_count = target_binarizer_final.shape[1]
		if test_flag == False:
			box.append(target_binarizer_final[:t_l])#training part 2
		else:
			box.append(target_binarizer_final[t_l:])#test part 2
	if stand_flag != 2:
		#target_other = min_max_scaler.fit_transform(target.T[target_filter == 0].T[:t_l])
		target_other = min_max_scaler.fit_transform(target.T[target_filter == 0].T)
		if test_flag == False:
			target_other = min_max_scaler.transform(target.T[target_filter == 0].T[:t_l])	
		else:
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
			target_other = preprocessing.scale(target.T[target_filter == 0].T[:t_l])
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

def column_picker(data_matrix, column_to_pick):
	"""to pick columns by ids"""
	import numpy as np
	target = np.array(data_matrix)
	assert max(column_to_pick) < target.shape[1]
	assert np.array(column_to_pick).min() >= 0
	target_step_1 = filter(lambda (x,y):x in column_to_pick, enumerate(target.T))
	return np.vstack([x[1].T for x in target_step_1]).T

def column_rearrange_num(data_matrix, new_order):
	import numpy as np
	target = np.array(data_matrix)
	(m, n) = target.shape
	assert n == len(set(new_order))
	assert n == np.array(new_order).max() + 1
	assert np.array(new_order).min() >= 0
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

def my_binarizer(*args):
	from sklearn import preprocessing
	import numpy as np
	assert len(args) == 2
	threshold = args[1]
	binarizer = preprocessing.Binarizer(threshold)
	return binarizer.transform(np.array(args[0]))

if __name__ == '__main__':
	pass

