#coding:utf8
"""
====================
Hcluster sample code with scikit 
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-10-27"""

__all__ = ['']

import sys, time
sys.path.append('..')
import numpy as np
from sklearn.cluster import Ward
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
#Hcluster
start = time.time()
ward = Ward(n_clusters = n).fit(cma)
label = ward.labels_
stop = time.time()
res = []
for i in xrange(len(label)):
        res.append((label[i],''.join(f[i]).decode('utf-8')))
res.sort()
for item in res:
        print item[0], item[1]
print stop-start
