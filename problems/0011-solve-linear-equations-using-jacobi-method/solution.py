import torch

def solve_jacobi(A, b, n) -> torch.Tensor:
    """
    Solve Ax = b using the Jacobi iterative method for n iterations.
    A: (m,m) tensor; b: (m,) tensor; n: number of iterations.
    Returns a 1-D tensor of length m, rounded to 4 decimals.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)
    # Your implementation here
    A_t = torch.as_tensor(A, dtype= torch.float)
    b_t = torch.as_tensor(b, dtype= torch.float)

    x = torch.zeros_like(b_t)

    for _ in range(n):
        x_new = torch.zeros_like(x)
        for i in range(len(b_t)):
            s = torch.sum(A_t[i] * x) - A_t[i, i]*x[i]
            x_new[i] = (b_t[i] - s)/ A_t[i, i]
        
        x = x_new
    return torch.round(x * 10000)/10000
