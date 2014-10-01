import numpy as np
import theano
import theano.tensor as T
from theano import shared

def basic():
	x = T.dvector('x')
	y = x**2
	f = theano.function(inputs=[x], outputs=y)
	a = np.array([1.,2.,5.])
	print f(a)

def derivation():
	x = T.dvector('x')
	y = x**2
	z = T.sum(x**2 + y)
	gz = T.grad(cost=z, wrt=x)
	f = theano.function(inputs=[x], outputs=gz)
	a = np.array([1.,2.,3.])
	print f(a)

def shared_v():
	

if __name__ == '__main__':
	#basic()
	derivation()
