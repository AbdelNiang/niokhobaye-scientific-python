"""Return each observation's nearest centroid using vectorized distances."""

import numpy as np


def assign_nearest_centroids(
    samples: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    if samples.ndim != 2 or centroids.ndim != 2:
        raise ValueError("samples and centroids must be two-dimensional")
    if samples.shape[1] != centroids.shape[1] or centroids.shape[0] == 0:
        raise ValueError("feature counts must match and centroids cannot be empty")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(centroids)):
        raise ValueError("samples and centroids must contain only finite values")

    dtype = np.result_type(samples.dtype, centroids.dtype, np.longdouble)
    samples = np.asarray(samples, dtype=dtype)
    centroids = np.asarray(centroids, dtype=dtype)

    deltas = samples[:, None, :] - centroids[None, :, :]
    squared_distances = np.einsum("ijk,ijk->ij", deltas, deltas, optimize=True)
    return np.argmin(squared_distances, axis=1)


def main() -> None:
    samples = np.array([[1, 1], [1, 2], [8, 8], [9, 8]], dtype=float)
    centroids = np.array([[1, 1.5], [8.5, 8]], dtype=float)
    assignments = assign_nearest_centroids(samples, centroids)
    print(f"Cluster assignments: {assignments}")


if __name__ == "__main__":
    main()