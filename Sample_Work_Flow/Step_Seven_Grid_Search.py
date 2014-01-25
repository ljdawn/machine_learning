if __name__ == '__main__':
	from sklearn import svm, grid_search, datasets
	iris = datasets.load_iris()
	parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100, 1000]}
	svr = svm.SVC()
	clf = grid_search.GridSearchCV(svr, parameters)
	clf.fit(iris.data, iris.target)
	print clf.grid_scores_