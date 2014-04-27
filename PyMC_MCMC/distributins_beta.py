from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
figsize(9, 6)

import scipy.stats as stats
from scipy.stats import beta

import json
import matplotlib
s = json.load(open("style/bmh_matplotlibrc.json"))
matplotlib.rcParams.update(s)

trials = [(0.1,0.1),(1.0,1.0),(2.0,3.0),(8.0,4.0)]
x = np.linspace(0, 1, 40)

for (k, (a, b)) in enumerate(trials):
	#sx = plt.subplot(len(trials), 2, k*2 + 1)
	sx = plt.subplot(1, 1, 1)
	rv = beta(a = a, b = b)
	y = rv.pdf(x)
	plt.plot(x, y, label="Alpha: %0.1f Beta: %0.1f" % (a, b))
	leg = plt.legend()
	leg.get_frame().set_alpha(0.4)
	plt.autoscale(tight=True)
	
plt.suptitle("beta distribution", y=1.01, fontsize=14)
plt.tight_layout()
plt.show()