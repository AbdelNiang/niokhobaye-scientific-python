"""Small, deterministic scientific Python utilities.

This module intentionally stays narrow: it contains three validated methods that
are useful for a Master's portfolio in optimization and scientific computing:

- ridge regression by solving a linear system,
- lasso regression by coordinate descent,
- gradient descent with an Armijo line search.
"""

from __future__ import annotations

import numpy as np


def _as_2d_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    return X


def ridge_regression(X: np.ndarray, y: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    """Solve the ridge regression problem min ||X beta - y||^2 + lam ||beta||^2.

    The closed-form solution is obtained by solving the linear system
    (X^T X + lam I) beta = X^T y.
    """
    X = _as_2d_matrix(X)
    y = np.asarray(y, dtype=float)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have incompatible shapes")
    if lam < 0:
        raise ValueError("lam must be non-negative")

    identity = np.eye(X.shape[1], dtype=float)
    return np.linalg.solve(X.T @ X + lam * identity, X.T @ y)


def soft_threshold(z: float, threshold: float) -> float:
    """Soft-thresholding used in coordinate descent for the lasso."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if z > threshold:
        return z - threshold
    if z < -threshold:
        return z + threshold
    return 0.0


def lasso_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    max_iter: int = 5000,
    tol: float = 1e-8,
) -> np.ndarray:
    """Estimate a lasso solution by cyclic coordinate descent.

    The model is min 0.5 ||X beta - y||^2 + lam ||beta||_1.
    """
    X = _as_2d_matrix(X)
    y = np.asarray(y, dtype=float)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have incompatible shapes")
    if lam < 0:
        raise ValueError("lam must be non-negative")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    n_samples, n_features = X.shape
    beta = np.zeros(n_features, dtype=float)
    residual = y.copy()

    for _ in range(max_iter):
        previous = beta.copy()
        for j in range(n_features):
            Xj = X[:, j]
            rho = Xj @ (residual + beta[j] * Xj)
            denominator = np.dot(Xj, Xj)
            if denominator <= 0:
                beta[j] = 0.0
            else:
                beta[j] = soft_threshold(rho, lam) / denominator
            residual = y - X @ beta
        if np.max(np.abs(beta - previous)) <= tol:
            break

    return beta


def armijo_gradient_descent(
    objective,
    gradient,
    x0: np.ndarray,
    initial_step: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
    max_iter: int = 200,
    tol: float = 1e-8,
):
    """Minimize f(x) by descent with Armijo backtracking.

    Returns (x_opt, history), where history is a list of iterates.
    """
    x = np.asarray(x0, dtype=float).copy()
    f0 = objective(x)
    step = float(initial_step)
    history = [x.copy()]

    for _ in range(max_iter):
        grad_x = np.asarray(gradient(x), dtype=float)
        if np.linalg.norm(grad_x) <= tol:
            break

        direction = -grad_x
        accepted = False
        local_step = step

        while local_step > 1e-12:
            candidate = x + local_step * direction
            if objective(candidate) <= f0 + c1 * local_step * np.dot(grad_x, direction):
                x = candidate
                f0 = objective(x)
                step = local_step
                history.append(x.copy())
                accepted = True
                break
            local_step *= rho

        if not accepted:
            raise RuntimeError("Armijo line search failed to find a valid step")

    return x, np.asarray(history, dtype=float)


__all__ = [
    "ridge_regression",
    "lasso_coordinate_descent",
    "armijo_gradient_descent",
]
