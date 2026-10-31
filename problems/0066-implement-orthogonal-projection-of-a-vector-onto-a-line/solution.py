import numpy as np

def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	v = np.asarray(v, dtype = float)
	L = np.asarray(L, dtype = float)

	projection = (np.dot(v, L) / np.dot(L, L))*L

	return np.round(projection, 3).tolist()