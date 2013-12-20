from sklearn import linear_model
LLM = linear_model.LogisticRegression

fn =  '../testing/data_glm'

for line in open(fn):
	print line