#coding:utf8
import numpy as np
import matplotlib.pyplot as plt
from math import exp

def read_data(path):
	res = []
	for line in open(path):
		res.append([float(x) for x in line.strip().split('   ')])
	return np.array(res)

def sigmoid(z):
	exp_m_z = [(exp(-zs) + 1.0) for zs in z]
	return [(1.0/res) for res in exp_m_z]

def Logistic_regression():
	x_path = 'ex4x.dat'
	y_path = 'ex4y.dat'
	x = read_data(x_path)
	y = read_data(y_path).T[0]
	[height, width] = x.shape

	intercept = np.ones(height)

	#iteratons Newton's Method
	theta = np.zeros(width + 1)
	x = np.append(intercept, x.T).reshape(width + 1, height)
	x = x.T
	z = x.dot(theta)
	h = np.array(sigmoid(z))

	#error??
	for i in xrange(100):
		#calculate grad
		grad = (1/float(height))*((h - y).T.dot(x))
		#calculate hessian
		A = x.T.dot(h).reshape(1, 3)
		B = (np.ones(len(h)) - h)
		C = B.dot(x).reshape(3, 1)
		H = C.dot(A)
		theta -= (1./H).dot(grad)
	print theta



def main():
	Logistic_regression()

if __name__ == '__main__':
	main()