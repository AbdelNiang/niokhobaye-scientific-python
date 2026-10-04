"""Small, deterministic scientific Python utilities.

This module intentionally stays narrow: it contains three validated methods that
are useful for a Master's portfolio in optimization and scientific computing:

- ridge regression by solving a linear system,
- lasso regression by coordinate descent,
- gradient descent with an Armijo backtracking line search.
"""

from __future__ import annotations

import numpy as np


def _as_2d_matrix(X: np.ndarray, *, name: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if X.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} must contain only finite values")
    return X


def _as_vector(y: np.ndarray, *, name: str = "y") -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim == 0:
        y = y.reshape(1)
    if y.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{name} must contain only finite values")
    return y


def ridge_regression(X: np.ndarray, y: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    """Solve the ridge regression problem

        minimize 0.5 * ||X beta - y||^2 + lam ||beta||^2.

    When lam = 0, the method falls back to a least-squares solve to keep the
    rank-deficient case numerically meaningful instead of invoking a singular
    linear system.
    """
    X = _as_2d_matrix(X, name="X")
    y = _as_vector(y, name="y")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have incompatible shapes")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lambda must be a finite non-negative scalar")

    if lam == 0.0:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta

    gram = X.T @ X
    rhs = X.T @ y
    identity = np.eye(X.shape[1], dtype=float)
    return np.linalg.solve(gram + lam * identity, rhs)


def soft_threshold(z: float, threshold: float) -> float:
    """Soft-thresholding used in coordinate descent for the lasso."""
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite non-negative scalar")
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

    The model is

        minimize 0.5 * ||X beta - y||^2 + lam ||beta||_1.

    The implementation keeps a residual updated incrementally to avoid repeated
    full matrix products at each coordinate update.
    """
    X = _as_2d_matrix(X, name="X")
    y = _as_vector(y, name="y")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have incompatible shapes")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lambda must be a finite non-negative scalar")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a finite non-negative scalar")
    if not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if X.shape[1] == 0:
        raise ValueError("X must have at least one feature")

    n_samples, n_features = X.shape
    beta = np.zeros(n_features, dtype=float)
    residual = y.copy()
    column_norms = np.sum(X * X, axis=0)

    for iteration in range(max_iter):
        previous = beta.copy()
        for j in range(n_features):
            if column_norms[j] <= 0.0:
                beta[j] = 0.0
                continue
            old_value = beta[j]
            rho = X[:, j] @ (residual + old_value * X[:, j])
            beta[j] = soft_threshold(rho, lam) / column_norms[j]
            residual -= (beta[j] - old_value) * X[:, j]

        if np.max(np.abs(beta - previous)) <= tol:
            return beta

    raise RuntimeError("Lasso coordinate descent reached max_iter without convergence")


def armijo_gradient_descent(
    objective,
    gradient,
    x0: np.ndarray,
    initial_step: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
    max_iter: int = 200,
    tol: float = 1e-8,
    *,
    return_status: bool = False,
):
    """Minimize f(x) by descent with Armijo backtracking.

    The function returns either (x_opt, history) or, when ``return_status`` is
    True, (x_opt, history, status) where status is one of ``converged`` or
    ``max_iter_reached``.
    """
    if not callable(objective):
        raise TypeError("objective must be callable")
    if not callable(gradient):
        raise TypeError("gradient must be callable")

    x = np.asarray(x0, dtype=float).copy()
    if x.ndim == 0:
        x = x.reshape(1)
    if not np.all(np.isfinite(x)):
        raise ValueError("x0 must contain only finite values")
    if not np.isfinite(initial_step) or initial_step <= 0.0:
        raise ValueError("initial_step must be a finite positive scalar")
    if not np.isfinite(c1) or not (0.0 < c1 < 1.0):
        raise ValueError("c1 must satisfy 0 < c1 < 1")
    if not np.isfinite(rho) or not (0.0 < rho < 1.0):
        raise ValueError("rho must satisfy 0 < rho < 1")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be a finite non-negative scalar")
    if not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")

    f0 = objective(x)
    if not np.isfinite(f0):
        raise ValueError("objective must evaluate to a finite value at x0")

    step = float(initial_step)
    history = [x.copy()]
    status = "max_iter_reached"
    max_backtracking_tries = 200

    for _ in range(max_iter):
        grad_x = np.asarray(gradient(x), dtype=float)
        if grad_x.shape != x.shape:
            raise ValueError("gradient shape is incompatible with x")
        if not np.all(np.isfinite(grad_x)):
            raise ValueError("gradient must contain only finite values")
        if np.linalg.norm(grad_x) <= tol:
            status = "converged"
            break

        direction = -grad_x
        accepted = False
        local_step = step

        for _ in range(max_backtracking_tries):
            candidate = x + local_step * direction
            candidate_value = objective(candidate)
            if not np.isfinite(candidate_value):
                raise ValueError("objective must evaluate to finite values along the line search")
            if candidate_value <= f0 + c1 * local_step * np.dot(grad_x, direction):
                x = candidate
                f0 = candidate_value
                step = local_step
                history.append(x.copy())
                accepted = True
                break
            local_step *= rho
            if local_step <= 1e-12:
                break

        if not accepted:
            raise RuntimeError("Armijo line search failed to find a valid step")

    history_array = np.asarray(history, dtype=float)
    if return_status:
        return x, history_array, status
    return x, history_array


__all__ = [
    "ridge_regression",
    "lasso_coordinate_descent",
    "armijo_gradient_descent",
]
