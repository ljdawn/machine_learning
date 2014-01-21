import pymc as pm
with pm.Model() as model:
	parameter = pm.Exponential("poisson_param",  1)
	data_generator = pm.Poisson("data_generator", parameter)
	data_plus_one = data_generator + 1
print data_plus_one