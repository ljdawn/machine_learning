from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
figsize(11, 5)

import scipy.stats as stats
from scipy.stats import gamma

import json
import matplotlib
s = json.load(open("style/bmh_matplotlibrc.json"))
matplotlib.rcParams.update(s)

trials = [(1,2.0),(2,2.0),(3,2.0),(5,1.0),(9,0.5)]
x = np.linspace(0, 20, 200)

for (k, (ke, si)) in enumerate(trials):
	#sx = plt.subplot(len(trials), 2, k*2 + 1)
	sx = plt.subplot(1, 2, 1)
	rv = gamma(a = ke, scale = si)
	y = rv.pdf(x)
	plt.plot(x, y, label="K: %0.1f S: %0.1f" % (ke, si))
	leg = plt.legend()
	leg.get_frame().set_alpha(0.4)
	plt.autoscale(tight=True)
	sx = plt.subplot(1, 2, 2)
	y = rv.cdf(x)
	plt.plot(x, y, label="K: %0.1f S: %0.1f" % (ke, si))
	plt.fill_between(x, 0, y, color="#348ABD", alpha=0.1)
	leg = plt.legend()
	leg.get_frame().set_alpha(0.4)
	plt.autoscale(tight=True)




plt.suptitle("normal distribution", y=1.01, fontsize=14)
plt.tight_layout()
plt.show()