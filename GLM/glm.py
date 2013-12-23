from sklearn import linear_model
import numpy as np

LLM = linear_model.LogisticRegression()

fn =  '../testing/data_glm'

X_pre = []
y_pre = []

for line in open(fn):
	X_c = map(float, line.strip().split(' ')[:-1])
	y_c = float(line.strip().split(' ')[-1])
	X_pre.append(X_c)
	y_pre.append(y_c)
X = np.array(X_pre)
y = np.array(y_pre)

Res = LLM.fit(X,y)

print Res.predict(X)
print y 





#t = [1,2,3,4,5,6,7,8]
#tm  = np.array(t)
#print tm.reshape(2,4)