#!/usr/bin/env python
#coding:utf8

import sys

def read_inputs(file, sep):
	for line in file:
		yield line.strip().split(sep)

def main(sep = '\t', m = 722774):
	data = read_inputs(sys.stdin, sep)
	for line in data:
		[I, J, i, j, v] = line
		for l in xrange(m):
			print '%s\t%s\t%d\t%s\t%s\t%s\t%s' %(I, J, l+1, 'O', i, j, v)
			#print '%d\t%s\t%s\t%s\t%s\t%s\t%s' %(l, J, I, 'T', j, i, v)
			print '%d\t%s\t%s\t%s\t%s\t%s\t%s' %(l+1, J, I, 'T', j, i, v)


main(sep = ' ', m = 5)
		
