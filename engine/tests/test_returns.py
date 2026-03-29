"""Unit tests for return helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.constants import TRADING_DAYS_PER_YEAR
from utils.returns import (
    annualized_return_from_daily_mean,
    cagr_from_price_series,
    daily_returns_from_prices,
    portfolio_daily_returns,
)


def test_daily_returns_simple():
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 99.0]},
        index=pd.date_range("2020-01-01", periods=3, freq="B"),
    )
    r = daily_returns_from_prices(prices)
    assert len(r) == 2
    assert pytest.approx(r["A"].iloc[0], rel=1e-6) == 0.10
    assert pytest.approx(r["A"].iloc[1], rel=1e-6) == -0.10


def test_annualized_from_daily_mean_zero():
    ar = annualized_return_from_daily_mean(0.0)
    assert ar == 0.0


def test_annualized_from_daily_mean_rejects_nan():
    with pytest.raises(ValueError):
        annualized_return_from_daily_mean(np.nan)


def test_cagr_two_year_doubling():
    idx = pd.date_range("2020-01-01", periods=3, freq="YE")
    s = pd.Series([100.0, 100.0, 200.0], index=idx)
    cagr, years = cagr_from_price_series(s)
    assert years > 0
    assert pytest.approx(cagr, rel=0.02) == (2.0 ** (1 / years) - 1)


def test_portfolio_daily_returns_weights():
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    rets = pd.DataFrame(
        {
            "A": [0.10, -0.05, 0.02],
            "B": [0.0, 0.10, 0.0],
        },
        index=dates[1:],
    )
    w = np.array([0.5, 0.5])
    p = portfolio_daily_returns(rets, w, symbols_order=["A", "B"])
    assert pytest.approx(p.iloc[0]) == 0.05
