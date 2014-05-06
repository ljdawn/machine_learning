import numpy as np
import scipy as sp
import math
import json
import matplotlib
import pymc as pm
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
	return 1.0/(1+np.exp(-x))

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
	return theta

def MCMC(x, y):
	x = x - x.mean()
	beta = pm.Uniform("beta", -1, 1)

	@pm.deterministic
	def p(t=x, beta=beta):
		return 1.0 / (1. + np.exp(-t*beta))

	observed = pm.Bernoulli("bernoulli_obs", p, value=y, observed=True)
	model = pm.Model([observed, beta])

	map_ = pm.MAP(model)
	map_.fit()
	mcmc = pm.MCMC(model)
	mcmc.sample(120000, 100000, 2)

	beta_samples = mcmc.trace('beta')[:, None]

	figsize(12.5, 6)

	plt.plot()
	plt.hist(beta_samples, histtype='stepfilled', bins=35, alpha=0.85,label=r"beta", color="#7A68A6", normed=True)
	plt.legend()
	plt.show()

def main():
	da = load_data()
	#pic_1 = show_data(da)
	#pic_1.show()
	print da
	theta = GD(da[:, 0], da[:, 1])
	print theta
	MCMC(da[:, 0], da[:, 1])

if __name__ == '__main__':
	main()

