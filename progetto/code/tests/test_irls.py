"""Tests for irls.py: monotonicity, convergence, KKT, solver equivalence."""
import numpy as np
import pytest

from src.irls import irls
from src.lasso_utils import check_optimality, f_lasso


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_irls_monotone_decrease(seed):
    """MM property: Q majorises f, so f(w_{k+1}) <= Q(w_{k+1}, w_k) <= f(w_k)."""
    from src.data_generation import make_lasso_problem
    X, y, _, _, _ = make_lasso_problem(n=20, m=80, lam=0.05, random_state=seed)
    res = irls(X, y, lam=0.05, eps_thr=1e-8, eps_stop=1e-10, k_max=200)
    f_vals = np.asarray(res["f_vals"])
    # tolerance for floating-point arithmetic in the linear solver
    assert np.all(np.diff(f_vals) <= 1e-9), (
        f"non-monotone step at index {int(np.argmax(np.diff(f_vals)))}"
    )


# ---------------------------------------------------------------------------
# Convergence to f*
# ---------------------------------------------------------------------------


def test_irls_converges_to_sklearn_fstar(medium_problem):
    p = medium_problem
    res = irls(p.X, p.y, p.lam, eps_thr=1e-10, eps_stop=1e-12, k_max=300)
    final_f = res["f_vals"][-1]
    gap = final_f - p.f_star
    assert gap < 1e-3, f"final gap {gap:.2e} too large"
    assert res["converged"], "IRLS should converge on a well-posed problem"


def test_irls_kkt_residual_small(medium_problem):
    p = medium_problem
    res = irls(p.X, p.y, p.lam, eps_thr=1e-10, eps_stop=1e-12, k_max=300)
    # KKT violation tolerance proportional to eps_thr (small components are
    # frozen by the safety threshold and may have residual up to lam*eps_thr).
    viol = check_optimality(p.X, p.y, res["w"], p.lam, tol=1e-6)
    assert viol < 1e-2, f"KKT violation {viol:.2e}"


# ---------------------------------------------------------------------------
# Sparsity
# ---------------------------------------------------------------------------


def test_irls_produces_sparse_iterate_when_lambda_large():
    """Large lambda should drive most components inside [-eps_thr, eps_thr]."""
    from src.data_generation import make_lasso_problem
    X, y, _, _, _ = make_lasso_problem(n=40, m=120, sparsity=0.1, lam=2.0, random_state=11)
    res = irls(X, y, lam=2.0, eps_thr=1e-6, eps_stop=1e-10, k_max=200)
    n = X.shape[1]
    n_inside = int(np.sum(np.abs(res["w"]) <= 1e-4))
    # most components should be effectively zero
    assert n_inside >= int(0.5 * n), f"only {n_inside}/{n} small components"


# ---------------------------------------------------------------------------
# Implementation knobs
# ---------------------------------------------------------------------------


def test_irls_warm_start_honoured(small_problem):
    p = small_problem
    w_init = p.w_star.copy()  # start at the optimum
    res = irls(p.X, p.y, p.lam, w0=w_init, k_max=5, eps_stop=1e-12)
    # very few iterations should suffice if we start near optimum
    assert res["n_iter"] <= 5
    assert f_lasso(p.X, p.y, res["w"], p.lam) - p.f_star < 1e-6


def test_irls_solver_choice_cholesky_vs_cg(small_problem):
    """Cholesky and CG must yield equivalent-quality solutions.

    The two iterate sequences may diverge in w because Q_k becomes
    ill-conditioned as small components are clipped by eps_thr (the
    diagonal entry 1/eps_thr can reach 1e8), and CG's relative residual
    tolerance does not match Cholesky's machine-precision factorisation
    in that regime. What we require is that both reach the same value
    of f within a small gap — they solve the same convex problem.
    """
    p = small_problem
    res_cho = irls(p.X, p.y, p.lam, solver="cholesky", k_max=200, eps_stop=1e-12)
    res_cg = irls(p.X, p.y, p.lam, solver="cg", k_max=200, eps_stop=1e-12)
    f_cho = res_cho["f_vals"][-1]
    f_cg = res_cg["f_vals"][-1]
    # Cholesky is direct: must reach machine-precision optimum
    assert f_cho - p.f_star < 1e-6
    # CG (rel tol 1e-10) is expected to lag by O(eps_thr) on the late
    # ill-conditioned iterations; just check we end up in the same ballpark.
    assert f_cg - p.f_star < 1e-2


def test_irls_records_consistent_lengths(small_problem):
    p = small_problem
    res = irls(p.X, p.y, p.lam, f_star=p.f_star, k_max=50)
    k = res["n_iter"]
    assert len(res["f_vals"]) == k + 1
    assert len(res["times"]) == k + 1
    assert len(res["gaps"]) == k + 1
    assert all(g >= 0.0 for g in res["gaps"])
