#!/usr/bin/env python
#coding:utf8

from itertools import groupby
from operator import itemgetter
import sys

def read_mapper_output(file, sep = '\t'):
	for line in file:
		yield line.strip().split(sep)

def main(sep = '\t', w = 1000, h = 1000):
	data = read_mapper_output(sys.stdin)
	#work for local check
	#data = sorted(data, key=itemgetter(0,1))
	for (key, group) in groupby(data,itemgetter(0,1)):
		matrix_R = {}
		for rec in group:
			[I, J, i, j, v] = rec
			if (i,j) not in matrix_R:
				matrix_R[(i,j)] = int(v)
			else:
				matrix_R[(i,j)] += int(v)
		for key_x in matrix_R:
			#print '%s\t%s\t%s\t%s\t%s' %(key[0], key[1], key_x[0], key_x[1], matrix_R[key_x])
			print '%d\t%d\t%s' %((int(key[0])-1)*h+int(key_x[0]), (int(key[1])-1)*h+int(key_x[1]), matrix_R[key_x])

sep = '\t'
w = 2
h = 2			
main(sep, w, h)
