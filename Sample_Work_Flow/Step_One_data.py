import pylab as pl
from sklearn.datasets import load_digits

digits = load_digits()
print digits.data.shape

print digits.data[0]
print digits.data[1000]