#coding:utf8
import numpy as np
import matplotlib.pyplot as plt

def read_data(path):
	res = []
	for line in open(path):
		res.append([float(x) for x in line.strip().split('   ')])
	return np.array(res)

def Linear_regression_2():
	x_path = 'ex3x.dat'
	y_path = 'ex3y.dat'
	x = read_data(x_path)
	y = read_data(y_path).T[0]
	[height, width] = x.shape

	intercept = np.ones(height)

	#scale
	x_std = x.std(axis = 0)
	x_mean = x.mean(axis = 0)
	x = [(x_s - x_mean)/x_std for x_s in x]
	x = np.vstack(x)
	x = np.append(intercept, x.T).reshape(width + 1, height)

	alpha = 1

	#iteratons
	theta = np.zeros(width + 1)

	for i in xrange(100):
		theta -= alpha*(1/float(height))*((theta.dot(x) - y).T.dot(x.T))
	print theta

def main():
	Linear_regression_2()

if __name__ == '__main__':
	main()