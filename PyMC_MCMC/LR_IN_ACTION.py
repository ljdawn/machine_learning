import numpy as np
import scipy as sp
import math
import json
import matplotlib
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier

s = json.load(open("style/bmh_matplotlibrc.json"))
matplotlib.rcParams.update(s)

def load_data():
	_data = np.genfromtxt("data/challenger_data.csv", skip_header = 1, usecols = [1, 2], missing_values="NA", delimiter = ",")
	return _data[~np.isnan(_data[:, 1])]

def show_data(da):
	plt.scatter(da[:, 0], da[:, 1], s = 75, color = "k", alpha = 0.5)
	plt.yticks([0, 1])
	return plt

def sigmoid(x):
	return 1.0/(1+math.exp(-x))

def GD(x, y, alpha = 0.001, numIterations = 1000):
	x = x - x.mean()
	n = 1
	m = len(x)
	theta = n
	for i in xrange(0, numIterations):
		hypothesis = map(sigmoid, (x*theta))
		loss = hypothesis - y
		gradient = np.dot(x, loss) / m
		theta = theta - alpha * gradient
	#y_ = map(sigmoid, (x*theta))
	#for i in xrange(len(y_)):
	#	print y[i],
	#	print 1.0 if y_[i] >= 0.29 else 0.0
	return theta

def SGD():
	pass

def MCMC():
	pass

def main():
	da = load_data()
	#pic_1 = show_data(da)
	#pic_1.show()
	theta = GD(da[:, 0], da[:, 1])
	print theta
	x = np.linspace(50, 85, 36)
	x = x - x.mean()
	y = map(sigmoid,(x*theta))
	plt.plot(x, y)
	plt.show()

if __name__ == '__main__':
	main()

