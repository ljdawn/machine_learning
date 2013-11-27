import numpy as np
import scipy as sp
from numpy import *
from numpy.linalg import *

A = [[1,0,2,1],
[-1,2,1,3],
[1,2,5,5],
[2,-2,1,-2]]

MA = array(A)
print MA

MAT = MA.transpose()
print MAT

#invMA = inv(MA)
#print invMA

traceMA = trace(MA)
print traceMA

print eig(MA)