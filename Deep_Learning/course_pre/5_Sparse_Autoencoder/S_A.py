#coding:utf8
import numpy as np
import matplotlib.pyplot as plt
from math import exp

def sigmoid(z):
	exp_m_z = [(exp(-zs) + 1.0) for zs in z]
	return [(1.0/res) for res in exp_m_z]

def d_sigmoid(z):
	f = sigmoid(z)
	return f*(np.ones(len(f))-f)

def main():
	pass

if __name__ == '__main__':
	main()
	a = [1,2,3,4]
	print d_sigmoid(a)