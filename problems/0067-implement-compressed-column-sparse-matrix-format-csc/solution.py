import numpy as np

def compressed_col_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

	:param dense_matrix: List of lists representing the dense matrix
	:return: Tuple of (values, row indices, column pointer)
	"""
	values = []
	row_indices = []
	col_ptr = [0]

	rows = len(dense_matrix)
	cols = len(dense_matrix[0]) if rows > 0 else 0

	for j in range(cols):
		for i in range(rows):
			if dense_matrix[i][j] != 0:
				values.append(dense_matrix[i][j])
				row_indices.append(i)
		
		col_ptr.append(len(values))

	return values, row_indices, col_ptr

