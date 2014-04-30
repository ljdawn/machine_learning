import numpy as np

m_tran = np.array([[0.65, 0.28, 0.07],[0.15, 0.67, 0.18],[0.12, 0.36, 0.52]])

#m = np.array([0.21, 0.68, 0.11])
m = np.array([0.1, 0.8, 0.1])

for i in range(10):
	m =  m.dot(m_tran)
	print m