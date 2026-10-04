import numpy as np
import pytest

from scientific_methods import armijo_gradient_descent, lasso_coordinate_descent, ridge_regression
from examples.nearest_centroids import assign_nearest_centroids


def test_ridge_regression_matches_closed_form():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    beta_true = np.array([1.0, -1.5, 0.5, 0.0, 2.0])
    y = X @ beta_true + 0.05 * rng.normal(size=20)

    beta = ridge_regression(X, y, lam=1e-2)
    expected = np.linalg.solve(X.T @ X + 1e-2 * np.eye(X.shape[1]), X.T @ y)

    assert np.allclose(beta, expected, rtol=1e-10, atol=1e-10)


def test_lasso_coordinate_descent_reduces_loss():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]], dtype=float)
    y = np.array([1.0, -1.0, 0.5, 0.5], dtype=float)

    beta = lasso_coordinate_descent(X, y, lam=0.1, max_iter=5000, tol=1e-10)
    mse = np.mean((X @ beta - y) ** 2)
    assert np.isfinite(beta).all()
    assert mse < 0.25


def test_armijo_descent_reduces_objective():
    def f(x):
        return 0.5 * np.sum((x - np.array([1.0, -2.0])) ** 2)

    def grad(x):
        return x - np.array([1.0, -2.0])

    x0 = np.array([3.0, 4.0])
    x_opt, history = armijo_gradient_descent(f, grad, x0, initial_step=1.0, max_iter=200)

    assert f(x_opt) < f(x0)
    assert len(history) >= 2
    assert np.isfinite(x_opt).all()


def test_armijo_rejects_invalid_parameters():
    def f(x):
        return np.sum(x**2)

    def grad(x):
        return 2 * x

    x0 = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="initial_step"):
        armijo_gradient_descent(f, grad, x0, initial_step=0.0)
    with pytest.raises(ValueError, match="rho"):
        armijo_gradient_descent(f, grad, x0, rho=1.0)
    with pytest.raises(ValueError, match="c1"):
        armijo_gradient_descent(f, grad, x0, c1=1.0)


def test_lasso_rejects_invalid_inputs():
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    y = np.array([1.0, 2.0], dtype=float)

    with pytest.raises(ValueError, match="lambda"):
        lasso_coordinate_descent(X, y, lam=np.nan)
    with pytest.raises(ValueError, match="incompatible|shape"):
        lasso_coordinate_descent(X, np.array([1.0]), lam=0.1)


def test_ridge_handles_zero_lambda_and_rank_deficient_case():
    X = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=float)
    y = np.array([3.0, 0.0], dtype=float)
    beta = ridge_regression(X, y, lam=0.0)
    assert np.allclose(beta, np.array([3.0, 0.0]), atol=1e-10)


def test_nearest_centroids_preserves_large_scale_order():
    samples = np.array([[1e9]], dtype=np.longdouble)
    centroids = np.array([
        [np.longdouble(1e9) + 1],
        [np.longdouble(1e9)],
    ], dtype=np.longdouble)
    assert assign_nearest_centroids(samples, centroids)[0] == 1
