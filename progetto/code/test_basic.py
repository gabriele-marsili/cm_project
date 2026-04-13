"""
test_basic.py
-------------
Quick sanity checks for all modules.
Run from the code/ directory:  python test_basic.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import traceback

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


def check(name, cond, msg=''):
    status = PASS if cond else FAIL
    print(f"  {status} {name}" + (f": {msg}" if msg else ''))
    return cond


def run_tests():
    all_passed = True

    # ------------------------------------------------------------------
    print("\n=== lasso_utils ===")
    from src.lasso_utils import f_lasso, grad_smooth, subgradient_f, optimality_gap

    np.random.seed(0)
    m, n = 20, 5
    X = np.random.randn(m, n)
    y = np.random.randn(m)
    w = np.random.randn(n)
    lam = 0.1

    fval = f_lasso(X, y, w, lam)
    all_passed &= check("f_lasso non-negative", fval >= 0)

    g = grad_smooth(X, y, w)
    all_passed &= check("grad_smooth shape", g.shape == (n,))

    # Numerical gradient check
    eps_fd = 1e-5
    g_num = np.zeros(n)
    for i in range(n):
        e = np.zeros(n); e[i] = eps_fd
        g_num[i] = (f_lasso(X, y, w+e, lam) - f_lasso(X, y, w-e, lam)) / (2*eps_fd)
    # smooth gradient only: compare with grad_smooth
    all_passed &= check("grad_smooth numerical check",
                        np.allclose(g, g_num - lam * np.sign(w), atol=1e-4),
                        f"max err={np.max(np.abs(g - (g_num - lam*np.sign(w)))):.2e}")

    # ------------------------------------------------------------------
    print("\n=== linear_solvers ===")
    from src.linear_solvers import cholesky_solve, conjugate_gradient, solve_spd

    Q = X.T @ X + 0.5 * np.eye(n)
    b = X.T @ y
    w_cho = cholesky_solve(Q, b)
    all_passed &= check("cholesky_solve residual",
                        np.linalg.norm(Q @ w_cho - b) < 1e-10,
                        f"{np.linalg.norm(Q @ w_cho - b):.2e}")

    w_cg, k_cg = conjugate_gradient(Q, b)
    all_passed &= check("conjugate_gradient residual",
                        np.linalg.norm(Q @ w_cg - b) < 1e-8,
                        f"{np.linalg.norm(Q @ w_cg - b):.2e} in {k_cg} iters")

    all_passed &= check("cholesky vs CG",
                        np.allclose(w_cho, w_cg, atol=1e-6))

    # ------------------------------------------------------------------
    print("\n=== IRLS ===")
    from src.irls import irls
    from src.lasso_utils import f_lasso

    np.random.seed(42)
    m2, n2 = 50, 10
    X2 = np.random.randn(m2, n2)
    w_true = np.zeros(n2); w_true[:3] = [1.5, -0.8, 0.4]
    y2 = X2 @ w_true + 0.05 * np.random.randn(m2)

    res = irls(X2, y2, lam=0.1, eps_thr=1e-8, eps_stop=1e-8, k_max=300)
    all_passed &= check("IRLS converged", res['converged'])
    all_passed &= check("IRLS f decreasing",
                        all(res['f_vals'][i] >= res['f_vals'][i+1] - 1e-10
                            for i in range(len(res['f_vals'])-1)),
                        "monotone decrease")

    from src.lasso_utils import check_optimality
    viol = check_optimality(X2, y2, res['w'], lam=0.1)
    all_passed &= check("IRLS optimality condition", viol < 1e-4, f"violation={viol:.2e}")

    # ------------------------------------------------------------------
    print("\n=== Deflected Subgradient ===")
    from src.deflected_subgradient import deflected_subgradient

    from sklearn.linear_model import Lasso as SkLasso
    sk = SkLasso(alpha=0.1 * m2 / 2, fit_intercept=False, max_iter=10000, tol=1e-12)
    sk.fit(X2, y2)
    w_sk = sk.coef_
    f_star_test = f_lasso(X2, y2, w_sk, 0.1)

    res_d = deflected_subgradient(X2, y2, lam=0.1, i_max=3000, beta=1.0,
                                   delta0=0.1*f_star_test, rho=0.95,
                                   f_star=f_star_test)
    all_passed &= check("DSM record non-increasing",
                        all(res_d['f_bar'][i] >= res_d['f_bar'][i+1] - 1e-12
                            for i in range(len(res_d['f_bar'])-1)))
    final_gap = res_d['gaps'][-1] if res_d['gaps'] else float('nan')
    all_passed &= check("DSM gap < 0.1", final_gap < 0.1, f"gap={final_gap:.3e}")

    # ------------------------------------------------------------------
    print("\n=== ELM ===")
    from src.elm import ELM

    elm = ELM(d=10, p=50, activation='sigmoid', lam=0.1, random_state=0)
    m3 = 100
    X_raw = np.random.randn(m3, 10)
    y3    = np.random.randn(m3)

    X_hid = elm.transform(X_raw)
    all_passed &= check("ELM transform shape", X_hid.shape == (m3, 50))
    all_passed &= check("ELM sigmoid range", X_hid.min() > 0 and X_hid.max() < 1)

    elm.fit(X_raw, y3, solver='irls', eps_stop=1e-6, k_max=50)
    y_hat = elm.predict(X_raw)
    all_passed &= check("ELM predict shape", y_hat.shape == (m3,))
    all_passed &= check("ELM w fitted", elm.w is not None)

    # ------------------------------------------------------------------
    print("\n=== data_generation ===")
    from src.data_generation import make_lasso_problem, make_elm_problem

    X_d, y_d, w_t, f_s, w_s = make_lasso_problem(n=30, m=100, lam=0.1,
                                                    random_state=0)
    all_passed &= check("make_lasso_problem shapes",
                        X_d.shape == (100, 30) and len(y_d) == 100)
    all_passed &= check("f_star positive", f_s > 0)

    # ------------------------------------------------------------------
    print()
    if all_passed:
        print(f"{PASS} All tests passed.")
    else:
        print(f"{FAIL} Some tests FAILED.")
    return all_passed


if __name__ == '__main__':
    ok = run_tests()
    sys.exit(0 if ok else 1)
