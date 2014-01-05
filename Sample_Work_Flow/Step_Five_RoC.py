def my_report(y_true, y_pred, class_names = ['class 1', 'class 2']):
	#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
	from sklearn.metrics import classification_report
	return(classification_report(y_true, y_pred, target_names = class_names))

y_true = [0, 1, 0, 1, 0]
y_pred = [0, 0, 0, 1, 0]

print my_report(y_true,y_pred)