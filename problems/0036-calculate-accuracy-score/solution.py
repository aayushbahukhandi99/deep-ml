import numpy as np

def accuracy_score(y_true, y_pred):
	# Your code here
	y_true = np.asarray(y_true, dtype= float)
	y_pred = np.asarray(y_pred, dtype= float)
	return float(np.mean(y_true == y_pred))