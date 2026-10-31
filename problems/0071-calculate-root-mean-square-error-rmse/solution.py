import numpy as np

def rmse(y_true, y_pred):
	# Write your code here
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)

	if not np.array_equal(y_true.shape, y_pred.shape) or y_true.size == 0:
		raise ValueError("Arrays must have the same shape and cannot be empty")
	mse = np.mean((y_true - y_pred) ** 2)
	return round(np.sqrt(mse),3)