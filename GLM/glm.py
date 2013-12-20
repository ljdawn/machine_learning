from sklearn import linear_model
import numpy as np

LLM = linear_model.LogisticRegression

fn =  '../testing/data_glm'

for line in open(fn):
	print line.strip()




t = [1,2,3,4,5,6,7,8]
tm  = np.array(t)

print tm.reshape(2,4)