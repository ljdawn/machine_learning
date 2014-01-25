if __name__ == '__main__':
	from sklearn import svm, grid_search, datasets
	from sklearn import linear_model
	iris = datasets.load_iris()
	parameters = {'penalty':['l1', 'l2'],'C':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
	M = linear_model.LogisticRegression()
	clf = grid_search.GridSearchCV(M, parameters)
	clf.fit(iris.data, iris.target)
	print clf.grid_scores_
	print clf.best_estimator_