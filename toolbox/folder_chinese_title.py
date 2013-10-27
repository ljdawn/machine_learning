"""
====================
folder chinese title filter 
====================

A chinese title filter.
"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-10-27"""

__all__ = ['']

#coding:utf8

import os,re
cf = re.compile(u'[\u4e00-\u9fa5]+')

def keep_chinese_name(path):
	for name in os.listdir(path):
		if cf.findall(name.decode('utf-8')) == []:
			p = path.strip('/')+'/'+name
			os.remove(p)
