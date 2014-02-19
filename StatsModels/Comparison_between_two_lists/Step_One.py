"""
====================
data comparision
Two
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-02-19"""

__all__ = ['']
import pandas as pd
import numpy as np
import scipy as sp
import statsmodels.api as sm

def range_histogram(list_left, list_right):
	list_left = np.array(list_left)
	list_right = np.array(list_right)
	range_upper = max(list_left.max(), list_right.max())
	range_lower = min(list_left.min(), list_right.min())
	range_all = range_upper - range_lower
	return range(int(range_all + 2))

def view_histogram(list_left, list_right, range_all):
	(left_val, bar) = np.histogram(list_left, bins = range_all)
	(right_val, bar) = np.histogram(list_right, bins = range_all)
	return (left_val, right_val, bar)

def single_ols(X, y):
	return sm.OLS(y, X).fit().summary()

if __name__ == '__main__':
    l_path = 'data/wt_sample3.out'
    r_path = 'data/pred_sample_3'
    head_path = 'data/head.txt'
    clo_name = 'site_type'
    clo_an = 'chengdan_flag'    

    clo_n = [cle.strip() for cle in open(head_path, 'r').readlines()]
    f_l = pd.read_table(l_path, names = clo_n, sep = '\t')
    f_r = pd.read_table(r_path, names = clo_n, sep = '\t')

    f_l_c = f_l[clo_name]
    f_l_a = f_l[clo_an]
    f_r_c = f_r[clo_name]
    f_r_a = f_r[clo_an]
    print clo_name, ':', view_histogram(f_l_c, f_r_c, range_histogram(f_l_c, f_r_c))
    print single_ols(f_l_c, f_l_a)
    for item in f_r_c:
    	if item not in (0,1,2):
    		print item
    #print single_ols(f_r_c, f_r_a)