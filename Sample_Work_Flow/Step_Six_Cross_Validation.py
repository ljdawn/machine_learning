import random
import numpy as np
from sklearn import svm, datasets

def my_CV(datasets, labels, func, k):
	#http://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics
	from sklearn import cross_validation
	assert len(datasets) == len(labels)
	return cross_validation.cross_val_score(func, datasets, labels, cv = k)

def my_CV_kfold(Y, n_folds = 10):
	from sklearn.cross_validation import StratifiedKFold
	skf = StratifiedKFold(Y, n_folds = n_folds)
	for train, test in skf:
		yield (train, test)

if __name__ == '__main__':
	iris = datasets.load_iris()
	func = svm.SVC(kernel='linear', C = 1)
	#print my_CV(iris.data, iris.target, func, 4)
	Y = [0, 0, 0, 1, 1, 1, 0,1, 0, 0, 0]
	for result in my_CV_kfold(Y, 4):
		print result