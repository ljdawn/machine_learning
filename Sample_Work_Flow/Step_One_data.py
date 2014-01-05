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
pred = np.array(probas.predict(data_test))

print pred
print type(label_test)