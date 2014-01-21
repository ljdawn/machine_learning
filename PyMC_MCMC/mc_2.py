#http://blog.csdn.net/sunmenggmail/article/details/17153053
import pymc as pm
with pm.Model() as model:
	pcoin = pm.Uniform("pcoin", 0, 1)
	experiment = pm.Binomial("experiment", 20, pcoin)

print experiment, pcoin