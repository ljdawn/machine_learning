#coding:utf8
"""
====================
NMF 
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-11-05"""

__all__ = ['']

import sys
sys.path.append('..')
import numpy as np
from toolbox import text_manufactory
from toolbox import text_chinese_filter
from sklearn import mixture

#para#
tf_idf = False
#text file path
path = '../testing/d_text_2_matrix'
#reading text file
f = open(path,'r').readlines()


#text to list_list
list_f = text_manufactory.text_2_list(f) 

#clean list_list
clean_list_f = text_chinese_filter.text_clean_list(list_f, en = False)
#print clean_list_f
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
X = np.array(cma)
print X
g = mixture.GMM(n_components=2)
g.fit(X)

print g.means
