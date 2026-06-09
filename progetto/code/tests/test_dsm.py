"""Tests for deflected_subgradient.py: gamma, monotonicity of f_bar, convergence."""
import numpy as np
import pytest

from src.deflected_subgradient import deflected_subgradient, _optimal_gamma
from src.lasso_utils import f_lasso, subgradient_f


# Optimal deflection (white-box on the helper)


def test_optimal_gamma_in_unit_interval(rng):
    for _ in range(20):
        g = rng.standard_normal(15)
        d = rng.standard_normal(15)
        gamma = _optimal_gamma(g, d)
        assert 0.0 <= gamma <= 1.0


def test_optimal_gamma_when_g_equals_d():
    g = np.array([1.0, -2.0, 3.0])
    gamma = _optimal_gamma(g, g)
    assert gamma == 1.0


def test_optimal_gamma_is_argmin_of_norm(rng):
    g = rng.standard_normal(10)
    d = rng.standard_normal(10)
    gamma_opt = _optimal_gamma(g, d)
    # numerical argmin over a fine grid
    grid = np.linspace(0, 1, 1001)
    norms = np.array([np.linalg.norm(t * g + (1 - t) * d) ** 2 for t in grid])
    gamma_grid = grid[int(np.argmin(norms))]
    assert abs(gamma_opt - gamma_grid) < 2e-3


# Warm start


def test_dsm_default_init_is_cold(small_problem):
    """Default w0 = 0 (cold start). The OLS warm start is admissible but is
    expected to be passed explicitly by the caller (see chapter 3, §parameter
    calibration)."""
    p = small_problem
    res = deflected_subgradient(p.X, p.y, p.lam, i_max=0)
    # i_max=0 means no iterations; w_best is the starting point.
    assert np.allclose(res["w"], np.zeros(p.X.shape[1]), atol=1e-12)


def test_dsm_user_warmstart_honoured(small_problem):
    p = small_problem
    custom = np.ones(p.X.shape[1]) * 0.5
    res = deflected_subgradient(p.X, p.y, p.lam, w0=custom, i_max=0)
    assert np.allclose(res["w"], custom, atol=1e-12)


# Record value monotonicity


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dsm_record_non_increasing(seed):
    from src.data_generation import make_lasso_problem
    X, y, _, f_star, _ = make_lasso_problem(n=20, m=80, lam=0.1, random_state=seed)
    res = deflected_subgradient(
        X, y, lam=0.1, i_max=2000, beta=1.0,
        delta0=0.1 * (f_star + 1.0), rho=0.95, f_star=f_star,
    )
    f_bar = np.asarray(res["f_bar"])
    assert np.all(np.diff(f_bar) <= 1e-12), "record value must be non-increasing"


# Convergence to f*


@pytest.mark.slow
def test_dsm_record_converges_to_fstar(small_problem):
    p = small_problem
    res = deflected_subgradient(
        p.X, p.y, p.lam, i_max=10000, beta=1.0,
        delta0=0.1 * p.f_star + 1e-3, rho=0.95, f_star=p.f_star,
    )
    final_gap = res["f_bar"][-1] - p.f_star
    # DSM is sublinear O(1/eps^2); 1e-2 in 10k iters is realistic
    assert final_gap < 5e-2, f"final gap {final_gap:.3e}"


# Subgradient sanity


def test_dsm_subgradient_is_in_subdifferential(small_problem, rng):
    """At any w, subgradient_f returns g such that f(w') >= f(w) + <g, w'-w>
    for w' in a small neighbourhood (sub-gradient inequality
    interpretation). Test on a random direction."""
    p = small_problem
    w = rng.standard_normal(p.X.shape[1])
    g = subgradient_f(p.X, p.y, w, p.lam)
    fw = f_lasso(p.X, p.y, w, p.lam)
    for _ in range(10):
        d = rng.standard_normal(len(w))
        d /= np.linalg.norm(d)
        # take a tiny step
        for h in [1e-5, 1e-4, 1e-3]:
            w_prime = w + h * d
            lhs = f_lasso(p.X, p.y, w_prime, p.lam)
            rhs = fw + h * np.dot(g, d)
            # sub-gradient inequality with FD slack
            assert lhs >= rhs - 1e-6, f"subgrad inequality violated by {rhs - lhs:.2e}"
