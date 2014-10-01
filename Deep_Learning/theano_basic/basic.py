import numpy as np
import theano
import theano.tensor as T
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams

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
	n = shared(np.array([1.5, 2.5, 3.5]), name='n')
	print n.get_value()
	f = theano.function(inputs=[], outputs=[], updates={n: n * 2})
	f()
	print n.get_value()
	f()
	print n.get_value()

def random_num():
	rng = RandomStreams(1234)
	x = rng.normal((10,))
	y = rng.binomial((10,), p=0.5, n=1)
	z = x * y
	f = theano.function(inputs=[], outputs=[x, y, z])
	f()
	print x, y, z
	f()
	print x, y, z


if __name__ == '__main__':
	#basic()
	#derivation()
	#shared_v()
	random_num()
