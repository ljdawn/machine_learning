#coding:utf8
"""
====================
feature selection tool 
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-10-30"""

__all__ = ['']
import os

def feature(inpath, outpath = '',  grouppath = []):
	con = {}
	for path_name in grouppath:
		dict_key = path_name.rstrip('/').split('/')[-1]
		con[dict_key] = {}
		for root, dirs, files in os.walk(path_name):
			for fn in files:
				f = open(root+fn,'r').readlines()
				fl =[line.strip().split('\t') for line in f]
				for ll in fl:
					for word in ll:
						con[dict_key][word] = '<'+dict_key+'>'
	for item in con:
		print item
	f = open(f_ori,'r').readlines()
	fl = [line.strip().split(' ') for line in f]
	fo = open(f_out, 'w')
	for ll in fl:
		conl  = []
		for word in ll:
			for dic in con:
				if word in con[dic]:
					word = con[dic][word]
			conl.append(word)
		fo.write(' '.join(conl)+'\n')
	fo.close()

