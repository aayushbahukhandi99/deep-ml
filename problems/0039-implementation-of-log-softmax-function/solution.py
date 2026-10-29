import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
	scores = np.asarray(scores, dtype=float)
	shifted = scores - np.max(scores)
	return shifted - np.log(np.sum(np.exp(shifted)))