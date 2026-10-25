def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	# Your code here
	centroids = [list(c) for c in initial_centroids]
	for _ in range(max_iterations):
		clusters = [[] for _ in range(k)]

		for point in points:
			distances = [
				sum((x-c)**2 for x, c in zip(point, centroid))
				for centroid in centroids
			]
			cluster_idx  = distances.index(min(distances))
			clusters[cluster_idx].append(point)
		
		new_centroids = []
		for i in range(k):
			if clusters[i]:
				centroid = tuple(
					sum(point[d] for point in clusters[i])/len(clusters[i])
					for d in range(len(points[0]))
				)
				new_centroids.append(list(centroid))
			else:
				new_centroids.append(centroids[i])
		if new_centroids == centroids:
			break
		centroids = new_centroids
	
	final_centroids = [
		tuple(round(value, 4) for value in centroid)
		for centroid in centroids
	]


	return final_centroids