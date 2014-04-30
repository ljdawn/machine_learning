import pymc as pm
import numpy as np
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt

import json
import matplotlib
s = json.load(open("style/bmh_matplotlibrc.json"))
matplotlib.rcParams.update(s)


figsize(12.5, 7)
count_data = np.loadtxt("data/txtdata.csv")
n_count_data = len(count_data)
plt.bar(np.arange(n_count_data), count_data, color="#348ABD")
plt.xlabel("Time (days)")
plt.ylabel("count of text-msgs received")
plt.title("Did the user's texting habits change over time?")
plt.xlim(0, n_count_data);
plt.show()