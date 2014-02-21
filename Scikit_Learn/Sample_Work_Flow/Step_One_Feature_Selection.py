def my_FS(X, y):
	from sklearn.feature_selection import chi2
	return chi2(X, y)

if __name__ == "__main__":
	from sklearn import datasets
	from sklearn.cross_validation import train_test_split
	iris = datasets.load_iris()
	X, data_test, y, label_test = train_test_split(iris.data[0:100], iris.target[0:100], train_size = 0.75, test_size = 0.25)
	print my_FS(X, y)