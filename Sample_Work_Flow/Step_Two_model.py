from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC

digits = load_digits(2)
data_train, data_test, label_train, label_test = train_test_split(digits.data, digits.target, train_size = 0.75, test_size = 0.25)
#http://scikit-learn.org/dev/modules/generated/sklearn.cross_validation.train_test_split.html
#digits.data data
#digits.target label

estimator = LinearSVC(C=1.0)
#http://scikit-learn.org/dev/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
#http://scikit-learn.org/dev/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
estimator.fit(data_train, label_train)

label_predict = estimator.predict(data_test)

print label_predict