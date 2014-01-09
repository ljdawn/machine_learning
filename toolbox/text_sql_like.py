"""
====================
text sql like 
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2014-01-09"""

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

def text_pop_col(list_, col_2_pop = -1):
	list_ =	list(list_)
	for line in list_:
		line.pop(col_2_pop)
		yield line

def text_keep_col(list_, col_2_keep = [0]):
	list_ =	list(list_)
	for line in list_:
		line_2 = []
		map(line_2.append,[line[i] for i in col_2_keep])
		yield line_2

def text_comb_col(list1_, list2_):
	list1_ = list(list1_)
	list2_ = list(list2_)
	if len(list1_) != len(list2_):
		print 'wrong format!'
		exit(1)
	else:
		for i in xrange(len(list1_)):
			yield list1_[i]+list2_[i]

def line_filter_by_short_commen_col(fl, fr, col = 0, sep = '\t'):
	"""return lines in two files with the same value in the perticular colounm
	"""
	f1 = [l.strip().split(sep) for l in open(fl).readlines()]
	f2 = [l.strip().split(sep) for l in open(fr).readlines()]
	(fS, fL, flag) = (f1, f2, 0) if len(f1) <= len(f2) else (f2, f1, 1)
	uq = [key[col] for key in (line for line in fS)]
	rline = filter(lambda x:x[col] in uq, fL)
	return (fS, rline) if flag == 0 else (rline, fS)
	
	
