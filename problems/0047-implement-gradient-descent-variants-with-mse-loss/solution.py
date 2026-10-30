import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    # Your code here
    X = np.asarray(X, dtype= float)
    y = np.asarray(y, dtype= float)
    weights = np.asarray(weights, dtype=float).copy()

    m = len(y)

    for epoch in range(n_epochs):
        if method == 'batch':
            predictions = X @ weights
            errors = predictions - y 
            gradient = (2/m) * (X.T @ errors)
            weights -= learning_rate*gradient

        elif method=='stochastic':
            for i in range(m):
                xi = X[i]
                error = xi @ weights - y[i]
                gradient = 2*error*xi
                weights -= learning_rate * gradient

        elif method == 'mini_batch':
            for i in range(0, m, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                predictions = X_batch @ weights
                errors = predictions - y_batch
                gradient = (2 / len(y_batch)) * (X_batch.T @ errors)

                weights -= learning_rate*gradient

        else:
            raise ValueError("Invalid method")
    return weights


