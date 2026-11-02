import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V 

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    d_k = K.shape[1]

    scores = (Q @ K.T) / np.sqrt(d_k)

    scores = scores - np.max(scores, axis = 1, keepdims = True)
    attention_weights = np.exp(scores)
    attention_weights /= np.sum(attention_weights, axis = 1, keepdims = True)
    return attention_weights @ V

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seq_len, d_model = Q.shape

    if n_heads <= 0 or d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")

    d_head = d_model // n_heads
    outputs = []

    for h in range(n_heads):
        start = h * d_head
        end = start + d_head

        Q_h = Q[:, start:end]
        K_h = K[:, start:end]
        V_h = V[:, start:end]

        outputs.append(self_attention(Q_h, K_h, V_h))

    return np.concatenate(outputs, axis = 1)