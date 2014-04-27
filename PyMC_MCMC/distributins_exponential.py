from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
figsize(11, 5)

import scipy.stats as stats
from scipy.stats import expon

import json
import matplotlib
s = json.load(open("style/bmh_matplotlibrc.json"))
matplotlib.rcParams.update(s)

trials = [0.5,1.0,1.5,2.0]
x = np.linspace(0, 5, 25)

for k, l in enumerate(trials):
	#sx = plt.subplot(len(trials), 2, k*2 + 1)
	sx = plt.subplot(1, 2, 1)
	rv = expon(scale = 1/l)
	y = rv.pdf(x)
	plt.plot(x, y, label="Lambda: %0.1f" % l)
	#plt.fill_between(x, 0, y, color="#348ABD", alpha=0.1)
	leg = plt.legend()
	leg.get_frame().set_alpha(0.4)
	plt.autoscale(tight=True)
	#sx = plt.subplot(len(trials), 2, k*2 + 2)
	sx = plt.subplot(1, 2, 2)
	rv = expon(scale = 1/l)
	y = rv.cdf(x)
	plt.plot(x, y, label="Lambda: %0.1f" % l)
	plt.fill_between(x, 0, y, color="#348ABD", alpha=0.1)
	leg = plt.legend()
	leg.get_frame().set_alpha(0.4)
	plt.autoscale(tight=True)



plt.suptitle("exponential distribution", y=1.0, fontsize=14)
plt.tight_layout()
plt.show()