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

From the repository root, either install the local package or add the project root to `PYTHONPATH`.

```bash
python3 -m pip install -e .
python3 examples/nearest_centroids.py
python3 examples/optimization_methods_demo.py
```

The demo script also supports a direct path invocation from the repository root:

```bash
PYTHONPATH=. python3 examples/optimization_methods_demo.py
```

If the `python` alias is unavailable on your machine, use `python3` instead.

## Validation

The implementation is checked by lightweight deterministic tests:

```bash
python3 -m pytest -q
```

## Scope

This repository is intentionally narrow. It does not claim broader optimization
frameworks or advanced algorithms before they are implemented and validated.

## Author

**Niokhobaye Abdel**

M.Sc. student in Mathematics, interested in optimization, scientific computing,
and mathematical modeling.
