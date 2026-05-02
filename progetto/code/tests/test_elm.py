"""
Tests for src/elm.py — the ELM model wrapping IRLS / DSM.
"""
import numpy as np
import pytest

from src.elm import ELM


def test_elm_transform_sigmoid_range():
    rng = np.random.default_rng(0)
    elm = ELM(d=5, p=20, activation="sigmoid", random_state=0)
    X = rng.standard_normal((30, 5))
    H = elm.transform(X)
    assert H.shape == (30, 20)
    assert np.all((H > 0.0) & (H < 1.0))


def test_elm_transform_tanh_range():
    rng = np.random.default_rng(1)
    elm = ELM(d=5, p=20, activation="tanh", random_state=0)
    X = rng.standard_normal((30, 5))
    H = elm.transform(X)
    assert np.all((H > -1.0) & (H < 1.0))


def test_elm_transform_relu_non_negative():
    rng = np.random.default_rng(2)
    elm = ELM(d=5, p=20, activation="relu", random_state=0)
    X = rng.standard_normal((30, 5))
    H = elm.transform(X)
    assert np.all(H >= 0.0)


def test_elm_unknown_activation_raises():
    with pytest.raises(ValueError):
        ELM(d=5, p=10, activation="elu")


def test_elm_predict_before_fit_raises():
    elm = ELM(d=5, p=10)
    with pytest.raises(RuntimeError):
        elm.predict(np.zeros((1, 5)))


def test_elm_fit_irls_then_predict(elm_problem):
    e = elm_problem
    elm = ELM(d=10, p=80, activation="sigmoid", lam=e["lam"], random_state=2)
    # Force the same hidden weights as the fixture used
    elm.W1 = e["W1"]
    elm.fit(e["X_raw"], e["y"], solver="irls", eps_stop=1e-8, k_max=200)
    y_hat = elm.predict(e["X_raw"])
    assert y_hat.shape == e["y"].shape
    # solution should be within IRLS-typical tolerance of the reference
    from src.lasso_utils import f_lasso
    f_irls = f_lasso(e["X_hid"], e["y"], elm.w, e["lam"])
    assert f_irls - e["f_star"] < 1e-2


def test_elm_fit_dsm_then_predict(elm_problem):
    e = elm_problem
    elm = ELM(d=10, p=80, activation="sigmoid", lam=e["lam"], random_state=2)
    elm.W1 = e["W1"]
    elm.fit(e["X_raw"], e["y"], solver="dsm", i_max=2000)
    y_hat = elm.predict(e["X_raw"])
    assert y_hat.shape == e["y"].shape
    assert elm.w is not None


def test_elm_unknown_solver_raises(small_problem):
    p = small_problem
    elm = ELM(d=p.X.shape[1], p=10)
    with pytest.raises(ValueError):
        elm.fit(np.zeros((5, p.X.shape[1])), np.zeros(5), solver="bfgs")


def test_elm_sparsity_property():
    elm = ELM(d=5, p=20, activation="sigmoid", lam=1.0, random_state=0)
    elm.w = np.array([0.0, 0.0, 1e-10, 0.5] + [0.0] * 16)
    assert 0.0 <= elm.sparsity <= 1.0
    assert elm.n_active + int(elm.sparsity * 20) == 20
