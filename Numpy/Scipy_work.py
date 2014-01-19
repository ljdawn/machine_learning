#scipy 
#---Optimization and Minimization---
"""data fitting sample"""
import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b):
	return a * x + b

x = np.linspace(1, 10, 100)
y = func(x, 1, 2)
yn = y + 0.9 * np.random.normal(size=len(x))
popt, pcov = curve_fit(func, x, yn)
print pcov
#pcov, where the diagonal elements are the variances for each parameter.
"""Solutions to Functions"""