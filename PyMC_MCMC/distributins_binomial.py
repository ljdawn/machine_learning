from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
figsize(11, 9)

import scipy.stats as stats
from scipy.stats import binom

import json
import matplotlib
s = json.load(open("style/bmh_matplotlibrc.json"))
matplotlib.rcParams.update(s)

trials = [(10,0.1),(10,0.2),(10,0.5),(10,0.9)]
x = np.arange(-1, 12, 1)

for (k, (N, p)) in enumerate(trials):
	sx = plt.subplot(len(trials), 2, k*2 + 1)
	rv = binom(N, p)
	y = rv.pmf(x)
	plt.bar(x-0.2, y, width=0.4,label="N: %d, P: %0.2f" % (N, p), color = "#348ABD")
	plt.vlines(p*10, 0, 0.5, color="k", linestyles="--", lw=1)
	leg = plt.legend()
	leg.get_frame().set_alpha(0.4)
	plt.autoscale(tight=True)
	sx = plt.subplot(len(trials), 2, k*2 + 2)
	rv = binom(N, p)
	y = rv.cdf(x)
	plt.bar(x-0.2, y, width=0.4,label="N %d, P %0.1f" % (N, p), color = "#348ABD")
	plt.vlines(p*10, 0, 1.1, color="k", linestyles="--", lw=1)
	leg = plt.legend()
	leg.get_frame().set_alpha(0.4)
	plt.autoscale(tight=True)

plt.suptitle("binomial distribution", y=0.99, fontsize=14)
plt.tight_layout()
plt.show()