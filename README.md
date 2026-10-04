# Scientific Python

A small portfolio of reproducible scientific Python examples. The goal here is
clarity, numerical stability, and real validation rather than breadth without
proof.

## Included methods

This repository currently contains a limited but verified set of methods:

- ridge regression by solving a linear system,
- lasso regression by coordinate descent,
- gradient descent with an Armijo backtracking line search.

The nearest-centroid example remains available as a simple vectorized baseline.

## Run the examples

```bash
python examples/nearest_centroids.py
python examples/optimization_methods_demo.py
```

## Validation

The implementation is checked by lightweight deterministic tests:

```bash
python -m pytest -q
```

## Scope

This repository is intentionally narrow. It does not claim broader optimization
frameworks or advanced algorithms before they are implemented and validated.

## Author

**Niokhobaye Abdel**

M.Sc. student in Mathematics, interested in optimization, scientific computing,
and mathematical modeling.
