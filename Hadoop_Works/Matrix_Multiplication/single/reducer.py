#!/usr/bin/env python
#coding:utf8

from itertools import groupby
from operator import itemgetter
import sys

def read_mapper_output(file, sep = '\t'):
	for line in file:
		yield line.strip().split(sep)

def main(sep = '\t'):
	data = read_mapper_output(sys.stdin)
	for (key, group) in groupby(data,itemgetter(0,1)):
		Box_A ={}
		Box_B ={}
		(location_i, location_j) = key 
		for rec in group:
			#print rec
			cl = rec[2]
			ke = rec[3]
			va = float(rec[4])
			if cl == 'A':
				Box_A[ke]=va
			else:
				Box_B[ke]=va
		res = 0.0
	#	print Box_A, Box_B
		for n in Box_A.keys():
			if n in Box_B.keys():
				res += Box_A[n]*Box_B[n]
		print key, res
main()
