if __name__ == '__main__':
	from sklearn import svm, grid_search, datasets
	from sklearn import linear_model
	iris = datasets.load_iris()
	#print iris
	parameters = {'penalty':['l1', 'l2'],'C':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
	M = linear_model.LogisticRegression(class_weight = {0 : 1.0, 1 : 1.5, 2 : 0.5}, penalty = 'l2', C = 1,)
	clf = grid_search.GridSearchCV(M, parameters)
	clf.fit(iris.data, iris.target)
	#print clf.grid_scores_
	print clf.best_params_