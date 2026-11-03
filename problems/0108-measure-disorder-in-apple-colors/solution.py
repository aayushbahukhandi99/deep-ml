import numpy as np
def disorder(apples: list) -> float:
	"""
	Compute the disorder in a basket of apples.
	"""
	# Your code here
	apples = np.asarray(apples)

	_, counts = np.unique(apples, return_counts = True)
	probabilities = counts/len(apples)

	result = 1 - np.sum(probabilities**2)
	
	return round(float(result), 4)