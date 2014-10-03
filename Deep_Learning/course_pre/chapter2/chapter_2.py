#coding:utf8
import numpy as np
import matplotlib.pyplot as plt

def read_data(path):
	res = []
	for line in open(path):
		res.append(float(line.strip()))
	return np.array(res)

def Linear_regression():
	x_path = 'ex2x.dat'
	y_path = 'ex2y.dat'
	x = read_data(x_path)
	y = read_data(y_path)
	print x
	height = len(x)
	width = 1

	intercept = np.ones(height)
	x = np.append(intercept, x).reshape(width + 1, height)
	print x
	alpha = 0.07

	#show data
	#plt.scatter(x, y)
	#plt.show()

	#first iteration
	theta = np.zeros(width + 1)
	theta_1 = theta - alpha*(1/float(height))*((theta.dot(x) - y).T.dot(x.T))
	#print theta_1

	#1500 iterations GD
	for i in xrange(1500):
		theta -= alpha*(1/float(height))*((theta.dot(x) - y).T.dot(x.T))
	#print theta

	#predict age3.5 & age 7
	print theta.dot(np.array([1, 3.5]))
	print theta.dot(np.array([1, 7]))

def main():
	Linear_regression()

if __name__ == '__main__':
	main()