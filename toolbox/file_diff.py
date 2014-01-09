"""
====================
text sql like 
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2014-01-09"""

__all__ = ['']

#coding:utf8
def print_diff(f1, f2):
	assert len(f1) == len(f2)
	n = len(f1) = len(f2)
	for i in xrange(n):
		yield map(lambda x, y:0 if x==y else 1,f1[i],f2[2])
