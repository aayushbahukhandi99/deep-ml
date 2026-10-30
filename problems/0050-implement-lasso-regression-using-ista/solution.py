import numpy as np

def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft-thresholding operator element-wise.
    
    S(w, λ) = sign(w) * max(|w| - λ, 0)
    
    Args:
        w: Input array
        threshold: Threshold value λ
    
    Returns:
        Soft-thresholded array where:
        - Values with |w| > λ are shrunk toward zero by λ
        - Values with |w| ≤ λ become exactly zero
    """
    # Your code here
    return np.sign(w)*np.maximum(np.abs(w) - threshold, 0)

def l1_regularization_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
    """
    Implement Lasso Regression using ISTA (Iterative Shrinkage-Thresholding Algorithm).
    
    ISTA alternates between:
    1. Gradient step on MSE loss: w_temp = w - lr * gradient_mse
    2. Proximal step (soft-thresholding): w_new = soft_threshold(w_temp, lr * alpha)
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        alpha: L1 regularization strength
        learning_rate: Step size for gradient descent
        max_iter: Maximum iterations
        tol: Convergence tolerance on weight change
    
    Returns:
        tuple: (weights, bias)
    
    Note: The bias term is NOT regularized.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    
    # Your code here
    for _ in range(max_iter):
        y_pred = X @ weights + bias
        errors = y_pred - y

        grad_w = (2 / n_samples) * (X.T @ errors)
        grad_b = (2 / n_samples) * np.sum(errors)

        w_temp = weights - learning_rate*grad_w
        new_bias = bias - learning_rate*grad_b

        new_weights = soft_threshold(w_temp, learning_rate * alpha)

        if np.linalg.norm(new_weights - weights) < tol:
            weights = new_weights
            bias = new_bias
            break
        
        weights = new_weights
        bias = new_bias

    return weights, bias
