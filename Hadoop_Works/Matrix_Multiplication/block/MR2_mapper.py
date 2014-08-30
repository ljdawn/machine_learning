#!/usr/bin/env python
#coding:utf8

import sys

def read_inputs(file, sep):
	for line in file:
		yield line.strip().split(sep, 3)

def main(sep = '\t'):
	data = read_inputs(sys.stdin, sep)
	for line in data:
		[I, J, K, con] = line
		n = 0
		for rec in con.split(sep):
			if n == 0:
				rec_temp = []
				rec_temp.append(rec)
				n += 1
			elif n == 1:
				rec_temp.append(rec)
				n += 1
			elif n == 2:
				rec_temp.append(rec)
				print '%s\t%s\t%s\t%s\t%s' %(I, K, rec_temp[0], rec_temp[1], rec_temp[2])
				n = 0
			else:
				pass


main(sep = '\t')
		
