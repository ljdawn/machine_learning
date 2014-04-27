from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
figsize(11, 9)

import scipy.stats as stats
from scipy.stats import expon

import json
import matplotlib
s = json.load(open("style/bmh_matplotlibrc.json"))
matplotlib.rcParams.update(s)

trials = [1,2,5,9]
x = np.arange(-1, 18, 1)

for k, l in enumerate(trials):
	sx = plt.subplot(len(trials), 2, k*2 + 1)
	rv = poisson(l)
	y = rv.pmf(x)
	plt.bar(x-0.2, y, width=0.4,label="Lambda: %d" % l, color = "#348ABD")
	#plt.vlines(l, 0, 0.5, color="k", linestyles="--", lw=1)
	leg = plt.legend()
	leg.get_frame().set_alpha(0.4)
	plt.autoscale(tight=True)
	sx = plt.subplot(len(trials), 2, k*2 + 2)
	rv = poisson(l)
	y = rv.cdf(x)
	plt.bar(x-0.2, y, width=0.4,label="Lambda: %d" % l, color = "#348ABD")
	#plt.vlines(l, 0, 0.5, color="k", linestyles="--", lw=1)
	leg = plt.legend()
	leg.get_frame().set_alpha(0.4)
	plt.autoscale(tight=True)


plt.suptitle("exponential distribution", y=0.99, fontsize=14)
plt.tight_layout()
plt.show()