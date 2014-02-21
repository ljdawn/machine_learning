from sklearn import linear_model
import numpy as np

LLM = linear_model.LogisticRegression(tol = 1e-10, penalty = 'l1', C = 18)

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

print X
print y

Res = LLM.fit(X,y)

print Res.predict(X)
print Res.predict_proba(X)
print y 

print Res.coef_
#print Res.get_params()

#[[-0.12732452  5.38572702 -0.03768956  2.71767444  3.07732422]]
#[[-0.0800647   1.21144739 -0.01576925  1.07963777  0.18775257]]
#[[-0.12759553  5.39914591 -0.03771755  2.7191854   3.08301482]]

#t = [1,2,3,4,5,6,7,8]
#tm  = np.array(t)
#print tm.reshape(2,4)