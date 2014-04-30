import pymc as pm
import numpy as np
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt

import json
import matplotlib
s = json.load(open("style/bmh_matplotlibrc.json"))
matplotlib.rcParams.update(s)


figsize(12.5, 3.5)
np.set_printoptions(precision=3, suppress=True)
challenger_data = np.genfromtxt("data/challenger_data.csv", skip_header=1, usecols=[1, 2], missing_values="NA", delimiter=",")
# drop the NA values
challenger_data = challenger_data[~np.isnan(challenger_data[:, 1])]

# plot it, as a function of tempature (the first column)
print "Temp (F), O-Ring failure?"
print challenger_data

plt.scatter(challenger_data[:, 0], challenger_data[:, 1], s=75, color="k",
            alpha=0.5)
plt.yticks([0, 1])
plt.ylabel("Damage Incident?")
plt.xlabel("Outside temperature (Fahrenheit)")
plt.title("Defects of the Space Shuttle O-Rings vs temperature")
plt.show()