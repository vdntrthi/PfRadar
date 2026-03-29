"""Risk and covariance tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.exceptions import RiskModelError
from utils.risk import (
    annualized_volatility_from_daily,
    covariance_matrix_from_returns,
    estimate_daily_mu_cov,
    validate_covariance,
)


def test_volatility_positive():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0, 0.01, 500))
    vol = annualized_volatility_from_daily(x)
    assert vol > 0


def test_covariance_symmetric_psd():
    rng = np.random.default_rng(1)
    n, t = 3, 1000
    A = rng.standard_normal((t, n))
    df = pd.DataFrame(A, columns=list("ABC"))
    c = covariance_matrix_from_returns(df)
    assert c.shape == (3, 3)
    assert np.allclose(c, c.T)
    validate_covariance(c, 3, eps_pd=-1e-9)


def test_estimate_mu_cov():
    rng = np.random.default_rng(2)
    t, n = 200, 2
    r = rng.normal(0, 0.01, size=(t, n))
    df = pd.DataFrame(r, columns=["X.NS", "Y.NS"])
    mu, cov, syms = estimate_daily_mu_cov(df, ridge_epsilon=1e-8)
    assert len(syms) == 2
    assert mu.shape == (2,)
    assert cov.shape == (2, 2)


def test_volatility_fails_tiny_sample():
    with pytest.raises(RiskModelError):
        annualized_volatility_from_daily(np.array([0.01]))
