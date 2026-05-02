"""
Shared pytest fixtures for the CM Project 25 ML test suite.

Three reference problems are exposed:
  - small_problem  : (m=50,  n=10)  fast unit tests
  - medium_problem : (m=200, n=50)  algorithm convergence tests
  - elm_problem    : full ELM pipeline (m=300, d=10, p=80)

Each fixture returns a NamedTuple bundling X, y, lam, w_star, f_star.
"""
import os
import sys
from typing import NamedTuple

import numpy as np
import pytest

# Make the `src` package importable: tests/conftest.py -> code/
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)


class LassoProblem(NamedTuple):
    X: np.ndarray
    y: np.ndarray
    lam: float
    w_true: np.ndarray
    w_star: np.ndarray
    f_star: float


def _make(n: int, m: int, lam: float, sparsity: float, seed: int) -> LassoProblem:
    from src.data_generation import make_lasso_problem
    X, y, w_true, f_star, w_star = make_lasso_problem(
        n=n, m=m, sparsity=sparsity, lam=lam, random_state=seed
    )
    return LassoProblem(X, y, lam, w_true, w_star, f_star)


@pytest.fixture(scope="session")
def small_problem() -> LassoProblem:
    return _make(n=10, m=50, lam=0.1, sparsity=0.3, seed=0)


@pytest.fixture(scope="session")
def medium_problem() -> LassoProblem:
    return _make(n=50, m=200, lam=0.05, sparsity=0.2, seed=1)


@pytest.fixture(scope="session")
def elm_problem():
    """Full ELM pipeline reference problem."""
    from src.data_generation import make_elm_problem
    X_raw, X_hid, y, W1, w_true, f_star, w_star = make_elm_problem(
        d=10, p=80, m=300, sparsity=0.15, lam=0.05, random_state=2
    )
    return {
        "X_raw": X_raw, "X_hid": X_hid, "y": y, "W1": W1,
        "w_true": w_true, "w_star": w_star, "f_star": f_star,
        "lam": 0.05,
    }


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260502)
