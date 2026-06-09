"""
Tests for src/data_generation.py - synthetic problem builders.

Critical invariant: f_star must be the actual minimum of f over R^n, i.e.
no random perturbation of w_star produces a lower value.
"""
import numpy as np
import pytest

from src.data_generation import make_lasso_problem, make_elm_problem
from src.lasso_utils import f_lasso, check_optimality


def test_make_lasso_problem_shapes():
    X, y, w_true, f_star, w_star = make_lasso_problem(n=20, m=80, lam=0.1, random_state=0)
    assert X.shape == (80, 20)
    assert y.shape == (80,)
    assert w_true.shape == (20,)
    assert w_star.shape == (20,)
    assert f_star >= 0.0


def test_make_lasso_problem_fstar_is_local_minimum():
    """f_star must be the minimum: any perturbation of w_star should not lower f."""
    X, y, _, f_star, w_star = make_lasso_problem(n=20, m=80, lam=0.1, random_state=0)
    rng = np.random.default_rng(0)
    for _ in range(30):
        d = rng.standard_normal(len(w_star))
        d /= np.linalg.norm(d)
        for h in [1e-3, 1e-2, 1e-1]:
            f_perturbed = f_lasso(X, y, w_star + h * d, 0.1)
            assert f_perturbed >= f_star - 1e-8, (
                f"perturbation lowered f by {f_star - f_perturbed:.3e}"
            )


def test_make_lasso_problem_kkt_at_wstar():
    X, y, _, _, w_star = make_lasso_problem(n=20, m=80, lam=0.1, random_state=0)
    viol = check_optimality(X, y, w_star, 0.1, zero_tol=1e-6)
    assert viol < 1e-3, f"sklearn solution KKT violation {viol:.2e}"


def test_make_elm_problem_consistency():
    X_raw, X_hid, y, W1, w_true, f_star, w_star = make_elm_problem(
        d=8, p=40, m=150, sparsity=0.2, lam=0.05, random_state=3
    )
    # f_star must equal f at w_star (sanity)
    f_eval = f_lasso(X_hid, y, w_star, 0.05)
    assert abs(f_eval - f_star) < 1e-10
    # X_hid must equal sigma(X_raw @ W1.T)
    from src.elm import _ACTIVATIONS
    sigma = _ACTIVATIONS["sigmoid"]
    assert np.allclose(X_hid, sigma(X_raw @ W1.T))


def test_make_elm_problem_kkt_at_wstar():
    _, X_hid, y, _, _, _, w_star = make_elm_problem(
        d=8, p=40, m=150, sparsity=0.2, lam=0.05, random_state=3
    )
    viol = check_optimality(X_hid, y, w_star, 0.05, zero_tol=1e-6)
    assert viol < 1e-2, f"ELM sklearn solution KKT violation {viol:.2e}"
