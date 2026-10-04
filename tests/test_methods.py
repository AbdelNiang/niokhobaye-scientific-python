import numpy as np

from scientific_methods import armijo_gradient_descent, lasso_coordinate_descent, ridge_regression


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
