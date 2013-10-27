"""
====================
text manufactory 
====================

Convert plain text to many other formats (or stored as python object).
"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-10-27"""

__all__ = ['']

#coding:utf8
import cPickle as pickle
import numpy
from itertools import *

def text_2_list(text_file_reader,delimiter = '\t'):
	"""text_file_reader is A list of file like:

	text_file_reader = open(<file_path>,'r').readlines

	if you are using csv.reader, you can skip this function."""
	return imap(lambda x:x.strip().split(delimiter), text_file_reader)

def list_2_matirx(list_):
	"""list_ is A list of list of word.
	This function changes A list_ to A sparse matrix"""
	list_ = list(list_)
	word_set = set([])
	word_dict = {}
	[map(word_set.add,word) for word in list_]
	for (wid, wd) in enumerate(list(word_set)):
		word_dict[wd]=wid
	return (((map(lambda x,y:(x[y]), [word_dict]*len(line),line) for line in list_),len(word_set), len(list_), list_),word_dict)

def list_tfidf(matrix_):
	"""martix_ is A like list_2_matirx(list_)[0][0].
	word_2_id is stored in list_2_matirx(list_)[1] as A python dictionary.
	This function calculates the tfidf values of the entire documents
	and returns a python dictionary like {key(word):value(tfidf)}
	gensim* has to be imported"""
	from gensim import models
	rdic ={}
	matrix_ = list(matrix_)
	corpus = [map(lambda x:(x,1.0),line) for line in matrix_]
	print corpus
	tfidf = models.TfidfModel(corpus)
	for rec in tfidf.idfs:
		rdic[rec] = tfidf.idfs[rec]
	return rdic
			 
