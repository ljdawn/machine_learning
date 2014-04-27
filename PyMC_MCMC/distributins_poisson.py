from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
figsize(11, 9)

import scipy.stats as stats
from scipy.stats import poisson

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
#l^k e^-l / k!
#l = 5, k = 5 -> f = 5^5 e^-5 /5!
#l = 5, k = 4 -> f = 5^4 e^-5/4!
#-> 5  / 5  



plt.suptitle("poisson distribution", y=0.99, fontsize=14)
plt.tight_layout()
plt.show()