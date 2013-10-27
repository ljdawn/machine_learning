"""
====================
text sql like 
====================

Convert plain text to many other formats (or stored as python object).
"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-10-27"""

__all__ = ['']

#coding:utf8
from itertools import *

def text_groupby(list_, col = 0):
	list_ = list(list_)
	list_ = sorted(list_, key = lambda x:x[col])
	list_I = groupby(map(lambda x:x[col], list_))
	list_it = iter(list_)
	del list_
	c_col = imap(lambda x:x[0],list_I)
	redict = {}
	try:
                while True:
                        rdk = c_col.next()
			redict[rdk] = []
        except StopIteration:pass
        finally:del c_col
	try:
		while True:
			line = list_it.next()
			redict[line[col]].append(line)	
	except StopIteration:pass
	finally:del list_it
	for lk in redict:
		yield (lk, redict[lk])
	
def text_change_col(list_, from_col = 1, to_col = 0):
	list_ = list(list_)
	for line in list_:
		fr_l = line[from_col]
		l_line = len(line)
		line.pop(from_col)
		line = iter(line)
		line_2 = []
		for i in xrange(l_line):
			if i == to_col:
				line_2.append(fr_l)
				print fr_l,'added!'
			else:
				l = line.next()
				line_2.append(l)
				print 'add*'
		yield(line_2)
		
