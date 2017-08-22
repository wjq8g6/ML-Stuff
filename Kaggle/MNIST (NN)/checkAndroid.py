import matplotlib.pyplot as plt
import numpy as np

file = open("example.txt", "r+")
arr = np.loadtxt(file)
arr = np.reshape(arr, (28,28))
plt.imshow(arr)
plt.show()