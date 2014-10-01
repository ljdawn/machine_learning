import numpy as np
import theano
import theano.tensor as T

def basic():
	x = T.dvector('x')
	y = x**2
	f = theano.function(inputs=[x], outputs=y)
	a = np.array([1.,2.,5.])
	print f(a)

if __name__ == '__main__':
	basic()