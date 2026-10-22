import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    # Your code here
    B = A.T @ A

    theta = 0.5 * np.arctan2(2 * B[0,1], B[0,0] - B[1, 1])

    c, s = np.cos(theta), np.sin(theta)

    V = np.array([[c, -s],[s, c]])

    D = V.T @ B @ V 
    S = np.sqrt(np.maximum(np.diag(D), 0))

    idx = np.argsort(S)[::-1]
    S = S[idx]
    V = V[:, idx]

    U = np.zeros((2, 2))
    for i in range(2):
        if S[i] > 1e-10:
            U[:, i] = A @ V[:, i] / S[i]

    if S[0] > 1e-10 and S[1] <= 1e-10:
        U[:, 1] = [-U[1,0], U[0, 0]]
    elif S[0] <= 1e-10:
        U = np.eye(2)

    return U, S, V.T
    