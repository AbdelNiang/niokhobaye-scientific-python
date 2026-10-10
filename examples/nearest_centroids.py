"""Return each observation's nearest centroid using vectorized distances."""

import numpy as np


def assign_nearest_centroids(
    samples: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    samples = np.asarray(samples)
    centroids = np.asarray(centroids)
    if samples.ndim != 2 or centroids.ndim != 2:
        raise ValueError("samples and centroids must be two-dimensional")
    if samples.shape[1] != centroids.shape[1] or centroids.shape[0] == 0:
        raise ValueError("feature counts must match and centroids cannot be empty")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(centroids)):
        raise ValueError("samples and centroids must contain only finite values")

    dtype = np.result_type(samples.dtype, centroids.dtype, np.longdouble)
    samples = np.asarray(samples, dtype=dtype)
    centroids = np.asarray(centroids, dtype=dtype)
    assignments = np.empty(samples.shape[0], dtype=int)
    elements_per_pair = max(samples.shape[1], 1)
    pair_budget = max(1, 1_000_000 // elements_per_pair)
    sample_block_size = max(1, min(samples.shape[0] or 1, int(np.sqrt(pair_budget))))
    centroid_block_size = max(1, min(centroids.shape[0], pair_budget // sample_block_size))

    for sample_start in range(0, samples.shape[0], sample_block_size):
        sample_stop = min(sample_start + sample_block_size, samples.shape[0])
        sample_block = samples[sample_start:sample_stop]
        best_distances = np.full(sample_block.shape[0], np.inf, dtype=dtype)
        best_indices = np.zeros(sample_block.shape[0], dtype=int)

        for centroid_start in range(0, centroids.shape[0], centroid_block_size):
            centroid_stop = min(centroid_start + centroid_block_size, centroids.shape[0])
            centroid_block = centroids[centroid_start:centroid_stop]
            deltas = sample_block[:, None, :] - centroid_block[None, :, :]
            squared_distances = np.einsum("ijk,ijk->ij", deltas, deltas, optimize=True)
            local_indices = np.argmin(squared_distances, axis=1)
            local_distances = squared_distances[np.arange(sample_block.shape[0]), local_indices]
            improved = local_distances < best_distances
            best_distances[improved] = local_distances[improved]
            best_indices[improved] = centroid_start + local_indices[improved]

        assignments[sample_start:sample_stop] = best_indices
    return assignments


def main() -> None:
    samples = np.array([[1, 1], [1, 2], [8, 8], [9, 8]], dtype=float)
    centroids = np.array([[1, 1.5], [8.5, 8]], dtype=float)
    assignments = assign_nearest_centroids(samples, centroids)
    print(f"Cluster assignments: {assignments}")


if __name__ == "__main__":
    main()