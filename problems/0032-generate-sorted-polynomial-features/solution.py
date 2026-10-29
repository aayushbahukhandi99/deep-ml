import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    # ✏️  Your code here
    X = np.asarray(X)
    n, d = X.shape

    features = [np.ones((n, 1)), X]

    for deg in range(2, degree+1):
        for combo in combinations_with_replacement(range(d), deg):
            features.append(np.prod(X[:, combo], axis=1).reshape(n,1))
    
    result = np.hstack(features)
    result.sort(axis=1)

    return result