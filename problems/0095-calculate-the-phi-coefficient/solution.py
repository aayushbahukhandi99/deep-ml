import numpy as np
def phi_corr(x: list[int], y: list[int]) -> float:
	"""
	Calculate the Phi coefficient between two binary variables.

	Args:
	x (list[int]): A list of binary values (0 or 1).
	y (list[int]): A list of binary values (0 or 1).

	Returns:
	float: The Phi coefficient rounded to 4 decimal places.
	"""
	# Your code here
	x = np.asarray(x, dtype= float)
	y = np.asarray(y, dtype= float)
	tp = np.sum((x == 1) & (y == 1))
	tn = np.sum((x == 0) & (y == 0))
	fp = np.sum((x == 0) & (y == 1))
	fn = np.sum((x == 1) & (y == 0))

	numerator = tp * tn - fp * fn

	denominator = np.sqrt(
		(tp + fp) * (tp + fn) *
		(tn + fp) * (tn + fn)
	)

	if denominator == 0:
		return 0.0
	val = numerator/denominator
	return round(val,4)