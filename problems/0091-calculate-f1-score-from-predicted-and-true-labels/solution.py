import numpy as np
def calculate_f1_score(y_true, y_pred):
	"""
	Calculate the F1 score based on true and predicted labels.

	Args:
		y_true (list): True labels (ground truth).
		y_pred (list): Predicted labels.

	Returns:
		float: The F1 score rounded to three decimal places.
	"""
	# Your code here
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)
	tp = np.sum((y_pred == 1) & (y_true==1))
	tn = np.sum((y_pred == 0) & (y_true==0))
	fp = np.sum((y_pred == 1) & (y_true==0))
	fn = np.sum((y_pred == 0) & (y_true==1))

	precision = tp/(tp + fp) if (tp + fp) else 0
	recall = tp/(tp + fn) if tp+fn else 0

	f1 = 2*precision*recall/(precision + recall) if (precision+recall) else 0
	return float(round(f1,3))