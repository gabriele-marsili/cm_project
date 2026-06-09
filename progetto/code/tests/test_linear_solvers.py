"""
Tests for src/linear_solvers.py - Cholesky and Conjugate Gradient on SPD systems.
"""
import numpy as np
import pytest

from src.linear_solvers import cholesky_solve, conjugate_gradient, solve_spd


def _random_spd(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)  # well-conditioned


@pytest.mark.parametrize("n,seed", [(5, 0), (20, 1), (100, 2)])
def test_cholesky_residual(n, seed):
    Q = _random_spd(n, seed)
    b = np.random.default_rng(seed + 100).standard_normal(n)
    x = cholesky_solve(Q, b)
    assert np.linalg.norm(Q @ x - b) < 1e-9


@pytest.mark.parametrize("n,seed", [(5, 0), (20, 1), (100, 2)])
def test_cg_matches_cholesky(n, seed):
    Q = _random_spd(n, seed)
    b = np.random.default_rng(seed + 200).standard_normal(n)
    x_cho = cholesky_solve(Q, b)
    x_cg, _ = conjugate_gradient(Q, b, tol=1e-12, max_iter=4 * n)
    assert np.allclose(x_cho, x_cg, atol=1e-7)


def test_cg_finite_termination_in_n_steps_exact():
    """In exact arithmetic CG converges in <= n iterations on SPD systems."""
    n = 30
    Q = _random_spd(n, 42)
    b = np.random.default_rng(42).standard_normal(n)
    x, k = conjugate_gradient(Q, b, tol=0.0, max_iter=n)
    # at most n iterations (theory); practically often a bit less
    assert k <= n
    assert np.linalg.norm(Q @ x - b) < 1e-6


def test_solve_spd_dispatch():
    Q = _random_spd(15, 7)
    b = np.random.default_rng(7).standard_normal(15)
    a = solve_spd(Q, b, method="cholesky")
    c = solve_spd(Q, b, method="cg", tol=1e-12)
    assert np.allclose(a, c, atol=1e-7)


def test_solve_spd_unknown_method_raises():
    with pytest.raises(ValueError):
        solve_spd(np.eye(3), np.zeros(3), method="lu")
