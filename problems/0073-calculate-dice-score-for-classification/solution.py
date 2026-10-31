
import numpy as np

def dice_score(y_true, y_pred):
	# Write your code here
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)

	tp = np.sum((y_pred == 1) & (y_true == 1))
	fp = np.sum((y_pred == 1) & (y_true == 0))
	fn = np.sum((y_pred == 0) & (y_true == 1))
	denom = (2*tp + fp + fn)
	res = 2*tp / denom if denom > 0 else 0.0
	return round(res, 3)
