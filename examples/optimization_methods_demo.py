"""Small deterministic demonstration for the limited scientific Python portfolio."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scientific_methods import armijo_gradient_descent, lasso_coordinate_descent, ridge_regression


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4))
    beta_true = np.array([1.5, -0.8, 0.3, 2.0], dtype=float)
    y = X @ beta_true + 0.1 * rng.normal(size=30)

    beta_ridge = ridge_regression(X, y, lam=1e-2)
    beta_lasso = lasso_coordinate_descent(X, y, lam=0.2)

    def objective(x: np.ndarray) -> float:
        return 0.5 * np.sum((x - np.array([1.0, -2.0])) ** 2)

    def gradient(x: np.ndarray) -> np.ndarray:
        return x - np.array([1.0, -2.0])

    x_opt, history = armijo_gradient_descent(objective, gradient, np.array([3.0, 4.0]))

    print("Ridge coefficients:", np.round(beta_ridge, 4))
    print("Lasso coefficients:", np.round(beta_lasso, 4))
    print("Armijo solution:", np.round(x_opt, 4))
    print("Iterations used:", len(history) - 1)


if __name__ == "__main__":
    main()
