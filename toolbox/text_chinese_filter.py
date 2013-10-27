"""
====================
text chinese filter 
====================

A chinese filter.
"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-10-27"""

__all__ = ['']

#coding:utf8
from itertools import *
import re

cn_only = re.compile(u'[\u4e00-\u9fa5]+')
cn_p_en = re.compile(u'[\u4e00-\u9fa5a-zA-Z0-9]+')

def text_clean_list(list_, en = False):
	"""make A list consisted of chinese(cn+en) character only .""" 
	list_ = list(list_)
	if en == True:
                finder = cn_p_en
	else:
		finder = cn_only
	flist = imap(lambda x: finder.findall(' '.join(x).decode('utf-8')), list_)
	return filter(lambda x: x != [],flist)
