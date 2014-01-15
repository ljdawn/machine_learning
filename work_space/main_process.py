"""
====================
data processing 
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-01-15"""

__all__ = ['']

#coding:utf8
from tool_box import Step_Zero_Processing_v01 as SZ

config_path = 'config/'
column_list_fn = config_path + 'head'
column_to_use_fn = config_path + 'works'
data_path = 'data/'
#test_file = data_path + 'test'
test_file = '../wt_sample_2_201110.txt'

column_picker = SZ.column_picker
preprocess = SZ.preprocess

#ori_list - [] name/label
def get_list(fn):
	with open(fn) as column_list_f:
		return map(lambda x:x.strip(), column_list_f.readlines())
#new_list - [] num/filter
def get_new_list(base, filter_):
	return map(lambda (x, y):x, filter(lambda (x, y):y in filter_, [(x, y) for (x,y) in enumerate(base)]))
#ori_data_matrix
def get_data_matrix(fn, sep = '\t'):
	res = []
	with open(fn) as mat:
		return map(lambda x:x.strip().split(sep), mat.readlines())
#ori_list - dict name/label
def get_list_label(base):
	res = {}
	for num, name in enumerate(base):
		res[num] = name
	return res
#new_list - [] name/label
def get_new_label(colum_label_dict, filter_):
	return [colum_label_dict[x] for x in filter_]


#name
column_list = get_list(column_list_fn)
#name
column_to_use_list = get_list(column_to_use_fn)
#num* sorted
column_new_list = get_new_list(column_list, column_to_use_list)
#name
column_label_dict = get_list_label(column_list)
#name* sorted  
colum_new_label = get_new_label(column_label_dict, column_new_list)
#ori_matrix
ori_data_mat = get_data_matrix(test_file)
#new_matrix 
data_matrix = column_picker(ori_data_mat, column_new_list)
#deal with 'NULL' transfer to float
data_matrix_float = [map(lambda x:float(x) if x != 'NULL' else 0, line) for line in data_matrix.tolist()]
#print data_matrix_float
#train_data
discret_list = [0,1,2,3,7,8]
train_data = preprocess(data_matrix_float, stand_flag = 1, discret_list = discret_list)
print train_data

#print column_list
#print column_to_use_list
#print column_new_list
#print column_label_dict
print get_new_label(column_label_dict, column_new_list)
#print get_data_matrix(test_file)
#print column_picker(ori_data_mat, column_new_list)


#print '\t'.join(colum_new_label)
#print '\t'.join(map(str,data_matrix_float[4]))

