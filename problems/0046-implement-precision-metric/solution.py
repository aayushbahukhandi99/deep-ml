import numpy as np

def precision(y_true, y_pred):
	# Your code here
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)

	tp = np.sum((y_true == 1) & (y_pred == 1))
	fp = np.sum((y_true != 1) & (y_pred == 1))

	return float(tp/ (tp+fp)) if (tp+ fp) > 0 else 0.0
