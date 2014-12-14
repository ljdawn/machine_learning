import theano.tensor as T
from theano import function
from theano import shared


#function
x = T.dscalar()
y = T.dscalar()
z = x - y
rep = T.dscalar()
f = function([x, y], z)
f1 = function([x, rep], z, givens=[(y, rep)])

print f(2,5), f1(2,5)




