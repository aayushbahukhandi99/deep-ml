def determinant_4x4(matrix: list[list[int|float]]) -> float:
	# Your recursive implementation here
	if len(matrix) == 1:
		return matrix[0][0]

	det = 0.0

	for j in range(len(matrix)):
		minor = [
			row[:j] + row[j + 1:]
			for row in matrix[1:]
		]

		det += ((-1) ** j)*matrix[0][j] * determinant_4x4(minor)
	return float(det)