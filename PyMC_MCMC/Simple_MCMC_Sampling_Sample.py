from scipy.stats import norm, uniform
from matplotlib import pyplot as plt
import matplotlib
import json
import numpy as np
import math

s = json.load(open("style/bmh_matplotlibrc.json"))
matplotlib.rcParams.update(s)

def simple_gibbs(niter = 50, rho = 0.7):
	mean1, mean2  = 10, 20
	std1, std2 = 1, 1
	sd =np.sqrt(1 - pow(rho, 2))
	x, y = 0.0, 0.0
	for i in xrange(niter):
		x = norm.rvs(mean1 + rho*(y - mean2)/std2, std1*sd)#p(x|y) ~ Norm(*)
		y = norm.rvs(mean2 + rho*(x - mean1)/std1, std2*sd)#p(y|x) ~ Norm(*) u should derive it
		yield (x,y)

def bivariate_norm_pdf(x, y, rho = 0.99):
	mean1, mean2  = 10, 20
	std1, std2 = 1, 1
	eq_1 = 1/(2*math.pi*std1*std2*np.sqrt(1-pow(rho,2)))
	eq_2 = pow((x - mean1), 2)/pow(std1, 2) + pow((y - mean2), 2)/pow(std2, 2) - 2*rho*(x - mean1)*(y - mean2)/std1*std2
	eq_3 = np.exp(-1*eq_2/2*(1-pow(rho, 2)))
	return eq_1 * eq_3

def simple_MH(niter = 100, rho = 0.99):
	x = [0]*niter
	x[0] = 0.5
	for i in xrange(niter - 1):
		xn = norm.rvs(x[i],0.05)
		acp = min(1.0, norm.pdf(xn)/norm.pdf(x[i]))
		u = uniform.rvs(0,1)
		#print u
		if u < acp:
			x[i + 1] = xn
		else:
			x[i + 1] = x[i]
	return x	


def main():
	n = 100000
	r = 0.99
	x = []
	y = []
	arr = simple_gibbs(n, r)
	for data in arr:
		da = arr.next()
		x.append(da[0])
		y.append(da[1])

	plt.plot()
	plt.scatter(x, y, color="#348ABD", alpha=0.85)
	plt.show()
	#print bivariate_norm_pdf(10000,100000)

	#b = simple_MH(n)
	#print b
	#plt.hist(b[:], bins = 100)
	#plt.show()

	#c = simple_MH_2(n, r)
	#plt.scatter(c[0], c[1], color="#348ABD", alpha=0.85)
	#plt.show()
	#print np.array(c[0][:]).mean(), np.array(c[1][:]).mean()


if __name__ == '__main__':
	main()





