"""
====================
text weka 
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-10-27"""

__all__ = ['']

#coding:utf8
from itertools import *
import text_sql_like as tl

def python_2_weka(list_,label=0,query=1):
	list_ = list(list_)
	f1 = tl.text_keep_col(list_,[label,query])
	if label != 1:
		f1 = tl.text_change_col(f1,0,1)
	f2 = tl.text_pop_col(list_, label)
	return (f1,f2)
	
