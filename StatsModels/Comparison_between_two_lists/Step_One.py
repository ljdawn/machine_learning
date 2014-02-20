"""
====================
data comparision
Two
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-02-19"""

__all__ = ['']
import math
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

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
	X = sm.add_constant(X)
	return sm.OLS(y, X).fit().summary()

if __name__ == '__main__':
    l_path = 'data/wt_sample3.out'
    r_path = 'data/pred_sample_3'
    head_path = 'data/head.txt'
    clo_an = 'chengdan_flag'    
    clo_n = [cle.strip() for cle in open(head_path, 'r').readlines()]
    f_l = pd.read_table(l_path, names = clo_n, sep = '\t')
    f_r = pd.read_table(r_path, names = clo_n, sep = '\t')
    
    def work(clo_name, clo_an):
        f_l_c = f_l[clo_name]
        f_l_c = map(lambda x:x if not math.isnan(x) else -1, f_l_c)
        f_l_a = f_l[clo_an]
        f_r_c = f_r[clo_name]
        f_r_c = map(lambda x:x if not math.isnan(x) else -1, f_r_c)
        f_r_a = f_r[clo_an]
        data_l = np.hstack((np.array([f_l_c]).T, np.array([f_l_a]).T))
        data_r = np.hstack((np.array([f_r_c]).T, np.array([f_r_a]).T))    
        data_l_i = imp.fit(data_l).transform(data_l).T
        data_r_i = imp.fit(data_r).transform(data_r).T 
        #f_l_c = np.array(data_l_i[0]).tolist()
        #f_r_c = np.array(data_r_i[0]).tolist()

        plt.subplot(211)
        n, bins_1, patches = plt.hist(f_l_c, range_histogram(f_l_c, f_r_c), normed=1, facecolor='green', alpha=0.5)
        y1 = mlab.normpdf(bins_1, np.array(f_l_c).mean(), np.array(f_l_c).var())
        plt.ylabel('Frequence')
        plt.subplots_adjust(left=0.15)
        plt.title(r'Histogram:'+ clo_name)
        plt.subplot(212)
        n, bins_2, patches = plt.hist(f_r_c, range_histogram(f_l_c, f_r_c), normed=1, facecolor='red', alpha=0.5)
        y2 = mlab.normpdf(bins_2, np.array(f_r_c).mean(), np.array(f_r_c).var())
        plt.ylabel('Frequence')
        plt.xlabel('Class/Values')
        plt.subplots_adjust(left=0.15)
        #plt.show()
        plt.savefig(clo_name+'.pdf')
        #plt.show()
        print clo_name, ':', 
        old, new,  base = view_histogram(f_l_c, f_r_c, range_histogram(f_l_c, f_r_c))
        #print single_ols(data_l_i[0], data_l_i[1])
        #print single_ols(data_r_i[0], data_r_i[1])
    #namelist = ['tenure_days','on_d14','on_m6','fn_d14','fn_m1','aan_d7','aan_m2','cn_m6s0','cn_m6s20','cn_m6s60',\
    #'cn_d7s120','cn_m6s180','cn_m1s180','cn_d7s180','dist_first_call','county_id','trade1_id','trade2_id','cust_type',\
    #'site_type','no_site_type','hint_source1','hint_source2','cust_source']
    namelist = ['chengdan_flag']
    for clo_name in namelist:
        work(clo_name, clo_an)