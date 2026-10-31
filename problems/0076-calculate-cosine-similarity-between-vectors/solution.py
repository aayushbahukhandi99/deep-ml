import numpy as np

def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	# Implement your code here
	v1 = np.asarray(v1)
	v2 = np.asarray(v2)

	denom = np.linalg.norm(v1) * np.linalg.norm(v2)

	if denom == 0:
		return 0.0

	cos_sim = np.dot(v1, v2)/denom

	return float(round(cos_sim, 5))