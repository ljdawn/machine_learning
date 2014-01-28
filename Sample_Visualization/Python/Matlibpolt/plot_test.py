import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 6)
t = [1,1.1,1.2,3,5,6.5]
n = map(int, x)

line, = plt.plot(n, map(lambda x : t[x], n), '--', linewidth=2)

plt.show()