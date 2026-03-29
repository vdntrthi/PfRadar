"""Live-data smoke test (network)."""

from __future__ import annotations

import numpy as np
import pytest

from models.exceptions import InsufficientHistoryError
from services.market_data import fetch_aligned_prices, normalize_indian_tickers
from services.optimizer import max_sharpe_weights, min_variance_weights
from services.report import build_full_report
from utils.returns import daily_returns_from_prices
from utils.risk import estimate_daily_mu_cov


def _skip_if_no_yahoo() -> None:
    """Call inside integration tests when yfinance returns empty (offline / rate limit)."""
    try:
        fetch_aligned_prices(["RELIANCE.NS"], min_history_trading_days=20)
    except InsufficientHistoryError as e:
        pytest.skip(f"Yahoo Finance data unavailable: {e}")


@pytest.mark.integration
def test_normalize_nse_aliases():
    assert normalize_indian_tickers(["RELIANCE", "TCS.NS"]) == ["RELIANCE.NS", "TCS.NS"]


@pytest.mark.integration
def test_fetch_reliance_tcs_infy():
    _skip_if_no_yahoo()
    prices, syms = fetch_aligned_prices(
        ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        min_history_trading_days=60,
    )
    assert len(syms) >= 2
    assert len(prices) >= 60
    rets = daily_returns_from_prices(prices[syms])
    mu, cov, used = estimate_daily_mu_cov(rets)
    assert len(used) == len(syms)
    w_mv = min_variance_weights(cov)
    w_ms = max_sharpe_weights(mu, cov, risk_free_annual=0.07)
    assert np.isclose(w_mv.sum(), 1.0, atol=1e-5)
    assert np.isclose(w_ms.sum(), 1.0, atol=1e-5)
    assert (w_mv >= -1e-8).all()
    assert (w_ms >= -1e-8).all()


@pytest.mark.integration
def test_build_full_report_json_safe():
    _skip_if_no_yahoo()
    rep = build_full_report(
        ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        random_portfolios=200,
        min_history_trading_days=60,
        plot_path=None,
    )
    assert "covariance_matrix" in rep
    assert "optimal_weights" in rep
    assert len(rep["symbols"]) >= 2
    assert rep["trading_days_per_year"] == 252
    import json

    json.dumps(rep)
