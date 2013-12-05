import numpy as np
import scipy as sp
from numpy import *
from numpy.linalg import *

A = [[1,-2,1],
[0,2,-8,],
[-4,5,9]]

B= [[0],[8],[-9]]

Ma = mat(A)
Mb = mat(B)

T = sovle(Ma,Mb)

print T
