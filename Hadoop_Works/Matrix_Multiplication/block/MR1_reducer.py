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
	data = sorted(data, key=itemgetter(0,1,2))
	for (key, group) in groupby(data,itemgetter(0,1,2)):
		matrix_O = {}
		matrix_T = {}
		for rec in group:
			[I, J, K, OT, i, j, v] = rec 
			if OT == 'O':
				matrix_O[(int(i), int(j))] = int(v)
			else:	
				matrix_T[(int(i), int(j))] = int(v)
		matrix_R = {}
		for key_O in matrix_O:
			for key_T in matrix_T:
				if key_O[1] == key_T[0]:
					if (key_O[0], key_T[1]) not in matrix_R:
						matrix_R[(key_O[0], key_T[1])] = matrix_O[key_O] * matrix_T[key_T]
					else:
						matrix_R[(key_O[0], key_T[1])] += matrix_O[key_O] * matrix_T[key_T]
		if matrix_R != {}:
			out_put = []
			for key_x in key:
				out_put.append(key_x)
			for item in matrix_R:
				out_put.append(str(item[0]))
				out_put.append(str(item[1]))
				out_put.append(str(matrix_R[item]))
			print '\t'.join(out_put)
			

main()
