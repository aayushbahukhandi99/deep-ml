import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	# Your code here
	errors = y_true - X @ w
	sum1 = np.mean(errors**2)
	sum2 = np.sum(w ** 2)

	return float(sum1 + alpha*sum2)

