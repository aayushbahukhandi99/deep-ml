import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
	"""
	Solve the system Ax = b using the Conjugate Gradient method.

	:param A: Symmetric positive-definite matrix
	:param b: Right-hand side vector
	:param n: Maximum number of iterations
	:param x0: Initial guess for solution (default is zero vector)
	:param tol: Convergence tolerance
	:return: Solution vector x
	"""
	# calculate initial residual vector
	A = np.asarray(A)
	b = np.asarray(b)

	x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype = float).copy()
	r = b - A@x
	p = r.copy()
	rs_old = np.dot(r, r)

	for _ in range(n):
		if np.sqrt(rs_old) < tol:
			break

		Ap = A @ p
		alpha = rs_old / np.dot(p, Ap)

		x = x + alpha*p 
		r = r - alpha*Ap

		rs_new = np.dot(r, r)
		beta = rs_new/rs_old

		p = r + beta*p
		rs_old = rs_new

	return x
