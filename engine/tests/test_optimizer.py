"""Optimizer unit tests (synthetic, no network)."""

from __future__ import annotations

import numpy as np
import pytest

from models.constants import TRADING_DAYS_PER_YEAR
from services.optimizer import (
    max_sharpe_weights,
    min_variance_weights,
    portfolio_mu_sigma_from_daily,
)


def test_min_variance_two_asset_diagonal_inverse_variance():
    """Diagonal Σ: long-only minimum variance is w_i ∝ 1/σ_i^2 (normalize)."""
    v0, v1 = 0.0004, 0.0001  # daily variances
    cov = np.diag([v0, v1])
    w = min_variance_weights(cov)
    inv = np.array([1.0 / v0, 1.0 / v1])
    w_exp = inv / inv.sum()
    assert w[0] + w[1] == pytest.approx(1.0)
    assert np.allclose(w, w_exp, atol=1e-4)


def test_max_sharpe_with_positive_spread():
    mu = np.array([0.001, 0.0002])
    cov = np.diag([0.0004, 0.0004])
    w = max_sharpe_weights(mu, cov, risk_free_annual=0.0)
    assert np.sum(w) == pytest.approx(1.0, abs=1e-6)
    assert w[0] > w[1]


def test_portfolio_mu_sigma_consistency():
    mu = np.array([0.0005, 0.0005])
    cov = np.eye(2) * 1e-4
    w = np.array([0.5, 0.5])
    m, s = portfolio_mu_sigma_from_daily(w, mu, cov)
    assert m == pytest.approx((1 + 0.0005) ** TRADING_DAYS_PER_YEAR - 1)
    assert s > 0
