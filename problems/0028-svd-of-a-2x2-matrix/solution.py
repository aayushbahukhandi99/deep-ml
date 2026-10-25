import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix.
    
    Args:
        A: 2x2 numpy array
    
    Returns:
        U: 2x2 orthogonal matrix (left singular vectors)
        s: 1D array of singular values
        V: 2x2 matrix (right singular vectors)
    """
    # Your code here
    A = np.array(A, dtype = float)
    B = A.T @ A
    theta = 0.5 * np.arctan2(2 * B[0,1], B[0, 0] - B[1, 1])

    c,s = np.cos(theta), np.sin(theta)
    V = np.array([[c, -s], [s, c]])

    D = V.T @ B @ V
    singular_values = np.sqrt(np.maximum(np.diag(D), 0))

    idx = np.argsort(singular_values)[::-1]
    singular_values = singular_values[idx]
    V = V[:, idx]

    U = np.zeros((2,2))
    for i in range(2):
        if singular_values[i] > 1e-10:
            U[:, i] = A @ V[:, i]/singular_values[i]
    
    if singular_values[0] > 1e-10 and singular_values[1] <= 1e-10:
        U[:, 1] = [-U[1, 0], U[0, 0]]
    elif singular_values[0] <= 1e-10:
        U = np.eye(2)
    

    return U, singular_values, V.T

    