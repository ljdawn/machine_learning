#!/usr/bin/env python
#coding:utf8

import sys

def read_inputs(file):
	for line in file:
		yield line.strip().split('\t')

def main(sep = '\t', m = 722774):
	data = read_inputs(sys.stdin)
	for line in data:
		[ia, ja, va] = line
		#print ia, ja, va
		[jb, ib, vb] = line
		#print ib, jb, vb
		for l in xrange(m):
			print ia+'\t'+str(l)+'\t'+'A'+'\t'+ja+'\t'+va
			print str(l)+'\t'+jb+'\t'+'B'+'\t'+ib+'\t'+vb
main()
		
