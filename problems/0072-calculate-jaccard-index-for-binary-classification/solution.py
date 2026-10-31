import numpy as np

def jaccard_index(y_true, y_pred):
	# Write your code here
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)

	intersection = np.sum((y_true == 1) & (y_pred == 1))
	union = np.sum((y_true == 1) | (y_pred == 1))

	result = intersection / union if union > 0 else 0.0
	
	return round(result, 3)
