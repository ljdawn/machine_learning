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
#pcov, where the diagonal elements are the variances for each parameter.
"""Solutions to Functions"""
#sample 1
from scipy.optimize import fsolve
import numpy as np
line = lambda x: x + 3
solution = fsolve(line, 0)
def findIntersection(func1, func2, x0):
	return fsolve(lambda x : func1(x) - func2(x), x0)

funky = lambda x : np.cos(x / 5) * np.sin(x / 2)
line = lambda x : 0.01 * x - 0.5

x = np.linspace(0,45,10000)
result = findIntersection(funky, line, [35])
#print(result, line(result))
#---Interpolation---
#---Integration---
#---Statistics---
from scipy import stats
sample = np.random.randn(100)
out0 = stats.normaltest(sample)
out1= stats.kstest(sample, 'norm')
out2 = stats.kstest(sample, 'wald')
print out2

"""
Array Creation
arange, array, copy, empty, empty_like, eye, fromfile, fromfunction, identity, linspace, logspace, mgrid, ogrid, ones, ones_like, r , zeros, zeros_like
Conversions
astype, atleast 1d, atleast 2d, atleast 3d, mat
Manipulations
array split, column stack, concatenate, diagonal, dsplit, dstack, hsplit, hstack, item, newaxis, ravel, repeat, reshape, resize, squeeze, swapaxes, take, transpose, vsplit, vstack
Questions
all, any, nonzero, where
Ordering
argmax, argmin, argsort, max, min, ptp, searchsorted, sort
Operations
choose, compress, cumprod, cumsum, inner, fill, imag, prod, put, putmask, real, sum
Basic Statistics
cov, mean, std, var
Basic Linear Algebra
cross, dot, outer, svd, vdot
"""