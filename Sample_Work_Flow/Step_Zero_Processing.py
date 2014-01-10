"""
====================
data processing 
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-01-10"""

__all__ = ['']

#coding:utf8
from operator import *

def my_Discretization(*args):
	assert len(args[2]) - len(args[1]) in (1, 0)
	assert len(args) == 4
	key = args[0]
	def inner(key_s):
		t_res = [0] * len(args[2])
		for i in xrange(len(args[1])):
			if args[-1](key_s, args[1][i]):
				#res = args[2][i]
				t_res[i] = 1
				break
			else:
				#res = args[2][-1]
				t_res[-1] = 1
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

if __name__ == '__main__':
	#1---
	box = [5, 4, 3, 2, 1]
	value = ['a','b','c','d','e']
	#ge --> '>=''
	op = ge
	to_prepare = [6, 5, 4, 3, 2, 1, 0]
	print my_Discretization(to_prepare, box, value, op)
	#2--
	train = [[ 1., -1.,  2.], [ 2.,  0.,  0.], [ 0.,  1., -1.]]
	to_prepare = [[-1.,  1., 0.]]
	print my_Standardization(train)
	print my_Standardization(train, to_prepare)
	#3--
	t = 2
	to_prepare = [0,1,2,3,4,5]
	print my_binarizer(to_prepare, t)


