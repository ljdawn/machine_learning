from scipy.stats import norm
from matplotlib import pyplot as plt
import matplotlib
import json
import numpy as np

s = json.load(open("style/bmh_matplotlibrc.json"))
matplotlib.rcParams.update(s)

def simple_gibbs(niter = 50, rho = 0.99):
	mean1, mean2  = 10, 20
	std1, std2 = 1, 1
	sd =np.sqrt(1 - pow(rho, 2))
	x, y = 0.0, 0.0
	for i in xrange(niter):
		x = norm.rvs(mean1 + rho*(y - mean2)/std2, std1*sd)
		y = norm.rvs(mean2 + rho*(x - mean1)/std1, std2*sd)
		yield (x,y)

def main():
	n = 10000
	r = 0.99
	x = []
	y = []
	arr = simple_gibbs(n, r)
	for i  in xrange(n):
		data = arr.next()
		x.append(data[0])
		y.append(data[1])

	plt.scatter(x, y, color="#348ABD", alpha=0.85)
	plt.show()

if __name__ == '__main__':
	main()





