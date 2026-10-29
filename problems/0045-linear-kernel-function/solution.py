import numpy as np

def kernel_function(x1, x2):
	# Your code here
	x1 = np.asarray(x1)
	x2 = np.asarray(x2)
	return np.dot(x1, x2)
