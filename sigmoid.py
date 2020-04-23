import matplotlib.pylab as plt
import numpy as np

x = np.arange(-8, 8, 0.1)
sigmoid = 1 / (1 + np.exp(-x))

plt.plot(x, sigmoid)
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.show()