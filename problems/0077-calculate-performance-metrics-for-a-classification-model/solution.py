import numpy as np
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
	# Implement your code here
	actual = np.asarray(actual, dtype= float)
	predicted = np.asarray(predicted, dtype= float)
	tp = np.sum((actual == 1) & (predicted == 1))
	tn = np.sum((actual == 0) & (predicted == 0))
	fp = np.sum((actual == 0) & (predicted == 1))
	fn = np.sum((actual == 1) & (predicted == 0))

	confusion_matrix = [[tp, fn], [fp, tn]]
	total = tp + fp + tn + fn
	accuracy = (tp + tn)/total if total else 0
	precision = tp / (tp + fp) if tp+fp else 0
	recall = tp / (tp + fn) if tp+fn else 0

	f1 = 2 * precision * recall/(precision + recall) if precision + recall else 0
	specificity = tn/(tn + fp) if tn + fp else 0
	negativePredictive = tn/(tn + fn) if tn + fn else 0

	return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
