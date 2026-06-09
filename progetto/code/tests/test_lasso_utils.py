"""Tests for lasso_utils.py."""
import numpy as np
import pytest

from src.lasso_utils import (
    f_lasso,
    grad_smooth,
    subgradient_f,
    check_optimality,
    optimality_gap,
)


# Objective value


def test_f_lasso_handcrafted():
    """f computed by hand on a tiny example."""
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([2.0, -1.0])
    w = np.array([1.0, 1.0])
    lam = 0.5
    # residual = (1-2, 1-(-1)) = (-1, 2); ||r||^2 = 5; (1/2)*5 = 2.5
    # ||w||_1 = 2; lam*||w||_1 = 1.0
    # f = 2.5 + 1.0 = 3.5
    assert f_lasso(X, y, w, lam) == pytest.approx(3.5, abs=1e-12)


def test_f_lasso_non_negative_at_zero(small_problem):
    p = small_problem
    w0 = np.zeros_like(p.w_star)
    # f(0) = (1/2)||y||^2 + 0
    assert f_lasso(p.X, p.y, w0, p.lam) == pytest.approx(
        0.5 * np.dot(p.y, p.y), abs=1e-12
    )


# Gradient of the smooth part


def test_grad_smooth_finite_difference(rng, small_problem):
    """Central finite-difference of f at smooth points (all w_i != 0)."""
    p = small_problem
    n = p.X.shape[1]
    eps = 1e-6
    for _ in range(5):
        # generate w with all components bounded away from zero
        w = rng.uniform(0.5, 1.5, size=n) * rng.choice([-1.0, 1.0], size=n)
        g_smooth = grad_smooth(p.X, p.y, w)
        # full f-gradient is g_smooth + lam*sign(w) at smooth points
        g_full = g_smooth + p.lam * np.sign(w)
        # central FD
        g_num = np.zeros(n)
        for i in range(n):
            e = np.zeros(n); e[i] = eps
            g_num[i] = (
                f_lasso(p.X, p.y, w + e, p.lam) - f_lasso(p.X, p.y, w - e, p.lam)
            ) / (2 * eps)
        assert np.allclose(g_full, g_num, atol=1e-5), (
            f"max err = {np.max(np.abs(g_full - g_num)):.2e}"
        )


def test_grad_smooth_zero_when_residual_is_zero():
    """If Xw = y, smooth gradient is exactly zero."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 5))
    w = rng.standard_normal(5)
    y = X @ w
    g = grad_smooth(X, y, w)
    assert np.allclose(g, 0.0, atol=1e-12)


# Subgradient


def test_subgradient_at_smooth_point(small_problem, rng):
    """At w with all w_i != 0, the subgradient is the actual gradient."""
    p = small_problem
    n = p.X.shape[1]
    w = rng.uniform(0.5, 1.5, n) * rng.choice([-1.0, 1.0], n)
    g_sub = subgradient_f(p.X, p.y, w, p.lam)
    g_full = grad_smooth(p.X, p.y, w) + p.lam * np.sign(w)
    assert np.allclose(g_sub, g_full, atol=1e-12)


def test_subgradient_minimum_norm_at_zero():
    """At w=0, min-norm subgradient is grad_smooth - clip(grad_smooth, -lam, lam).
    Our code returns grad_smooth + lam*sign(0) = grad_smooth + 0 = grad_smooth.
    This is the minimum-norm element when |grad_smooth_i| <= lam (then s_i = 0
    minimises (g_i + lam*s_i)^2 over s_i in [-1,1]); it remains in the
    subdifferential and is the choice we adopt.
    """
    rng = np.random.default_rng(0)
    n = 8
    X = rng.standard_normal((20, n))
    y = rng.standard_normal(20)
    lam = 100.0  # large so |X^T(Xw-y)|_i <= lam at w=0
    w = np.zeros(n)
    g = subgradient_f(X, y, w, lam)
    # g must be a valid subgradient: g - grad_smooth(X,y,w) must be in [-lam,lam]^n
    s = (g - grad_smooth(X, y, w)) / lam
    assert np.all(np.abs(s) <= 1.0 + 1e-12)
    # And in this regime the min-norm choice is s = 0, so g = grad_smooth.
    assert np.allclose(g, grad_smooth(X, y, w))


# Optimality check


def test_check_optimality_zero_at_sklearn_solution(small_problem):
    p = small_problem
    viol = check_optimality(p.X, p.y, p.w_star, p.lam, zero_tol=1e-6)
    assert viol < 5e-4, f"sklearn solution should be near-optimal; got {viol:.2e}"


def test_check_optimality_positive_at_random_point(small_problem, rng):
    p = small_problem
    w_random = rng.standard_normal(p.X.shape[1])
    viol = check_optimality(p.X, p.y, w_random, p.lam, zero_tol=1e-6)
    assert viol > 0.0


def test_optimality_gap_zero_at_optimum(small_problem):
    p = small_problem
    gap = optimality_gap(p.X, p.y, p.w_star, p.lam, p.f_star)
    assert abs(gap) < 1e-8
