#coding:utf8
"""
====================
DBSCAN sample code with scikit 
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-10-27"""

__all__ = ['']

import sys, time
sys.path.append('..')
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from toolbox import text_manufactory
from toolbox import text_chinese_filter

#para#
n = 3
tf_idf = True
#text file path
path = '../testing/d_text_2_matrix'
#reading text file
f = open(path,'r').readlines()


#text to list_list
list_f = text_manufactory.text_2_list(f) 

#clean list_list
clean_list_f = text_chinese_filter.text_clean_list(list_f, en = False)
print clean_list_f
#list_list to term vector matrix
((clean_M, term_count, sample_count, Di),word_dict) = text_manufactory.list_2_matirx(clean_list_f) 
clean_M = list(clean_M)

#tfidf
tfidf = text_manufactory.list_tfidf(clean_M)

#shape
cma = np.zeros((sample_count,term_count))
for i in xrange(len(clean_M)):
	for j in clean_M[i]:
		if tf_idf == False:
			cma[i][j] = 1
		else:
			cma[i][j] = tfidf[j]
#DBSCAN
start = time.time()
db = DBSCAN(eps=0.05, min_samples = n).fit(cma)
stop = time.time()
labels = db.labels_
for i in xrange(len(labels)):
        print labels[i],''.join(f[i]).decode('utf-8')
print stop-start
