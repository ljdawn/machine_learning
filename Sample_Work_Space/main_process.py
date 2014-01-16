"""
====================
data processing 
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-01-16"""

__all__ = ['']

#coding:utf8
from tool_box import Step_Zero_Processing_v01 as SZ

#---tool_box common functions---
column_picker = SZ.column_picker
column_rearrange_num = SZ.column_rearrange_num
column_get_label_num = SZ.column_get_label_num
preprocess = SZ.preprocess

#---main functions---
def get_list(fn):
	with open(fn) as column_list_f:
		return map(lambda x:x.strip(), column_list_f.readlines())
def get_new_list(base, filter_):
	return map(lambda (x, y):x, filter(lambda (x, y):y in filter_, [(x, y) for (x,y) in enumerate(base)]))
def get_data_matrix(fn, sep = '\t'):
	res = []
	with open(fn) as mat:
		return map(lambda x:x.strip().split(sep), mat.readlines())
def get_list_label(base):
	res = {}
	for num, name in enumerate(base):
		res[num] = name
	return res
def get_new_label(colum_label_dict, filter_):
	return [colum_label_dict[x] for x in filter_]
def get_One_col(fn, col = -1):
	with open(fn) as column_One:
		return map(lambda x:x.strip()[col], column_One.readlines())

#---config/data files---
config_path = 'config/'
column_list_fn_ori = config_path + 'head_ori'
column_list_fn_new = config_path + 'head_new'
column_to_use_fn = config_path + 'works'
data_path = 'data/'
data_file = data_path + 'test'
#test_file = '../wt_sample_2_201110.txt'

#---process start---
#<<step 1 -- rearrage data_matrix-- >>
#get ori_label, new_label
column_label_ori = get_list(column_list_fn_ori)
column_label_new = get_list(column_list_fn_new)
column_num_new = column_get_label_num(column_label_ori, column_label_new)
#get ori_matrix new matrix
data_matrix_ori = get_data_matrix(data_file)
data_matrix_new = column_rearrange_num(data_matrix_ori, column_num_new)
#get effective matrix
column_to_use_list = get_list(column_to_use_fn)
column_new_list = get_new_list(column_label_new, column_to_use_list)
data_matrix_eff = column_picker(data_matrix_new, column_new_list)
data_matrix_eff_float = [map(lambda x:float(x) if x != 'NULL' else 0, line) for line in data_matrix_eff.tolist()]

#<<step 2 -- preprossing data_matrix-- >>
#def column type
discret_list = [4,5,6,7,8,9]
train_data = preprocess(data_matrix_eff_float, stand_flag = 1, discret_list = discret_list)
print train_data[1]

#<<step 3 -- get flag-- >>
res_list = get_One_col(data_file)
print res_list
