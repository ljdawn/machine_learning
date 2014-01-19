import numpy as np
#---array-------------------------------------------------------------------------
#ndarray is ~25 faster then python loops.
#matrix can only will be two dimensional.

#array creation samples
"""single value"""
arr = np.array(100)
"""range values"""
arr = np.arange(0,100)
"""step values"""
arr = np.linspace(0,1,100)
arr = np.logspace(0, 1, 100, base=10.0)
"""0s, 1s"""
arr = np.zeros(5)
arr = np.ones(5)
cube = np.zeros((5,5,5)).astype(int) + 1

#---array can store other type data---
recarr = np.zeros((2,), dtype=('i4,f4,a10'))
toadd = [(1,2.,'Hello'),(2,3.,"World")]
recarr[:] = toadd

#---index and slicing---
"""np.where(), np.delete()"""
arr = np.arange(5)
index = np.where(arr > 2)
new_arr = np.delete(arr, index)
"""same with
index = arr > 2
print(index)
[False False True True True]
new_arr = arr[index]
"""
#---Boolean Statements-------------------------------------------------------------------------
"""as before"""
import numpy as np
import numpy.random as rand
a = rand.randn(100)
index = a > 0.2
b = a[index]

#---math-------------------------------------------------------------------------
#Linear Algebra