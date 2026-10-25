import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	weights = initial_weights.astype(float).copy()
	bias = float(initial_bias)
	labels = labels.reshape(-1)
	m = len(labels)
	mse_values = []

	for _ in range(epochs):
		z = features@weights + bias
		probabilities = 1 / (1 + np.exp(-z))

		errors = probabilities - labels
		mse = np.mean(errors **2)
		mse_values.append(mse)

		dz = 2 * errors * probabilities * (1-probabilities)
		dw = (features.T @ dz) /m 
		db = np.mean(dz)

		weights -= learning_rate * dw
		bias -= learning_rate*db
	
	updated_weights = np.round(weights, 4)
	updated_bias = np.round(bias, 4)
	mse_values = [round(value, 4) for value in mse_values]

	return updated_weights, updated_bias, mse_values