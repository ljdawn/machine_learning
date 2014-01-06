import random
import numpy as np
from sklearn import svm, datasets

def my_CV(datasets, labels, func, k):
	#http://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics
	from sklearn import cross_validation
	assert len(datasets) == len(labels)
	return cross_validation.cross_val_score(func, datasets, labels, cv = k)

if __name__ == '__main__':
	iris = datasets.load_iris()
	func = svm.SVC(kernel='linear', C=1)
	print my_CV(iris.data, iris.target, func, 4)
	