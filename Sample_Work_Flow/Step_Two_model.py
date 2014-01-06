import random
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.cross_validation import train_test_split
from sklearn.metrics import auc

iris = datasets.load_iris()

data_train, data_test, label_train, label_test = train_test_split(iris.data[0:100], iris.target[0:100], train_size = 0.75, test_size = 0.25)

classifier = svm.SVC(kernel='linear', probability=True)
probas = classifier.fit(data_train, label_train)
#http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
pred = np.array(probas.predict_proba(data_test))
pred_int = np.array(probas.predict(data_test))

print list(pred[:,1])
#pred[:,1]+pred[:,0] = 1
print list(pred_int)
print list(label_test)