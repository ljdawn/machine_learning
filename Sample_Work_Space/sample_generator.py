"""
====================
model 
sample generator
====================

"""
__author__ = """\n""".join(['Xuan Zhang'])

__version__ ="""2013-01-24"""

__all__ = ['']

#coding:utf8
from tool_box import Step_Six_Cross_Validation as SS
my_CV_kfold = SS.my_CV_kfold

if __name__ == '__main__':
	Y = [0, 0, 0, 1, 1, 1, 0,1, 0, 0, 0]
	for result in my_CV_kfold(Y, 4):
		print result