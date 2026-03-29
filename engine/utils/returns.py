"""
Return calculations from adjusted close prices.

Uses simple (pct_change) daily returns. Annualization uses TRADING_DAYS_PER_YEAR only.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from models.constants import TRADING_DAYS_PER_YEAR

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def daily_returns_from_prices(
    prices: pd.DataFrame,
    *,
    how: str = "any",
) -> pd.DataFrame:
    """
    Compute simple daily returns: P_t / P_{t-1} - 1.

    Parameters
    ----------
    prices
        Aligned adjusted close, columns = symbols.
    how
        `any` = drop rows with any NaN (full history required per date).
        `all` = drop only if all NaN in row.

    Returns
    -------
    DataFrame of daily returns, index aligned to return dates.
    """
    r = prices.sort_index().pct_change()
    if how == "any":
        return r.dropna(how="any")
    if how == "all":
        return r.dropna(how="all")
    raise ValueError("how must be 'any' or 'all'")


def annualized_return_from_daily_mean(
    mean_daily: float | np.floating,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Compound annualization: (1 + mu_d)^TDY - 1.

    Guards
    ------
    - If mean_daily is NaN, raises ValueError.
    """
    m = float(mean_daily)
    if not np.isfinite(m):
        raise ValueError("mean_daily must be finite")
    return float((1.0 + m) ** float(trading_days) - 1.0)


def cagr_from_price_series(
    prices: pd.Series,
    *,
    min_points: int = 2,
) -> tuple[float, float]:
    """
    CAGR from first to last price over the index time span.

    CAGR = (P_end / P_start)^(1/years) - 1
    years = (index[-1] - index[0]).days / 365.25

    Returns
    -------
    (cagr, years_elapsed). If years_elapsed <= 0 or insufficient data, returns (nan, years).
    """
    s = prices.dropna().sort_index()
    if len(s) < min_points:
        logger.warning("CAGR: insufficient points (%s)", len(s))
        return float("nan"), 0.0
    p0 = float(s.iloc[0])
    p1 = float(s.iloc[-1])
    if p0 <= 0 or p1 <= 0:
        logger.warning("CAGR: non-positive price boundary")
        return float("nan"), 0.0
    t0, t1 = s.index[0], s.index[-1]
    try:
        days = (t1 - t0).days
    except (TypeError, AttributeError):
        days = len(s)  # fallback for non-datetime index
    years = float(days) / 365.25
    if years <= 0:
        return float("nan"), years
    total_return_factor = p1 / p0
    cagr = float(total_return_factor ** (1.0 / years) - 1.0)
    return cagr, years


def portfolio_daily_returns(
    asset_returns: pd.DataFrame,
    weights: np.ndarray,
    *,
    symbols_order: list[str],
) -> pd.Series:
    """
    Daily portfolio simple returns: R_p,t = sum_i w_i * r_i,t.

    Parameters
    ----------
    asset_returns
        Columns must match symbols_order.
    weights
        Length n, sum to 1.
    symbols_order
        Column order for weights alignment.
    """
    if list(asset_returns.columns) != symbols_order:
        asset_returns = asset_returns[symbols_order]
    w = np.asarray(weights, dtype=float).reshape(1, -1)
    raw = (asset_returns.values * w).sum(axis=1)
    return pd.Series(raw, index=asset_returns.index, name="portfolio")
