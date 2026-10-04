"""Return each observation's nearest centroid using vectorized distances."""

import numpy as np


def assign_nearest_centroids(
    samples: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    if samples.ndim != 2 or centroids.ndim != 2:
        raise ValueError("samples and centroids must be two-dimensional")
    if samples.shape[1] != centroids.shape[1] or centroids.shape[0] == 0:
        raise ValueError("feature counts must match and centroids cannot be empty")

    sample_norms = np.sum(samples**2, axis=1, keepdims=True)
    centroid_norms = np.sum(centroids**2, axis=1)
    squared_distances = sample_norms + centroid_norms - 2 * samples @ centroids.T
    return np.argmin(squared_distances, axis=1)


def main() -> None:
    samples = np.array([[1, 1], [1, 2], [8, 8], [9, 8]], dtype=float)
    centroids = np.array([[1, 1.5], [8.5, 8]], dtype=float)
    assignments = assign_nearest_centroids(samples, centroids)
    print(f"Cluster assignments: {assignments}")


if __name__ == "__main__":
    main()