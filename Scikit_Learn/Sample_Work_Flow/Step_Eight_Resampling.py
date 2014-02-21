#http://scikit-learn.org/dev/modules/generated/sklearn.utils.resample.html
import numpy as np

X = [[1., 0.], [2., 1.], [0., 0.],[8., 4.], [5., -5], [7., 8.]]
y = np.array([0, 1, 2])

from sklearn.utils import resample
print resample(X, replace = False, n_samples = 5, random_state = 25)