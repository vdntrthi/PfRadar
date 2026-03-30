"""
CAPM (Capital Asset Pricing Model) utilities for Indian equities.

Fetches Nifty 50 as the market proxy, computes per-stock beta,
and derives CAPM-implied annualized expected returns.

Formula:
    E(Ri) = Rf + beta_i * (E(Rm) - Rf)

where:
    E(Rm) = annualized historical market return (Nifty 50)
    beta_i = Cov(Ri, Rm) / Var(Rm)   [computed from daily returns]
    Rf     = risk-free rate (India 10Y G-Sec proxy, default 7%)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from models.constants import DEFAULT_RISK_FREE_ANNUAL_IN, TRADING_DAYS_PER_YEAR
from models.exceptions import DataFetchError

logger = logging.getLogger(__name__)

_NIFTY_SYMBOL = "^NSEI"


def fetch_market_returns(
    start: date | None = None,
    end: date | None = None,
) -> pd.Series:
    start_s = start.isoformat() if start else (date.today() - timedelta(days=365 * 5)).isoformat()
    end_s = end.isoformat() if end else None

    try:
        df = yf.download(
            _NIFTY_SYMBOL,
            start=start_s,
            end=end_s,
            progress=False,
            threads=False,
            auto_adjust=True,
        )
        if df is None or df.empty or "Close" not in df.columns:
            raise DataFetchError("Empty response for ^NSEI")

        closes = df["Close"].copy()
        if isinstance(closes.index, pd.DatetimeIndex):
            closes.index = closes.index.tz_localize(None).normalize()

        returns = closes.astype(float).pct_change().dropna()
        returns.name = "market"
        logger.info("Fetched %d Nifty 50 daily returns", len(returns))
        return returns

    except Exception as e:
        logger.warning("Could not fetch Nifty 50 returns: %s — CAPM will be skipped", e)
        return pd.Series(dtype=float, name="market")


def compute_betas(
    stock_daily_returns: pd.DataFrame,
    market_daily_returns: pd.Series,
) -> dict[str, float]:
    aligned = stock_daily_returns.join(
        market_daily_returns.rename("__market__"), how="inner"
    ).dropna()

    if len(aligned) < 20:
        logger.warning(
            "Only %d overlapping rows for beta — defaulting all betas to 1.0",
            len(aligned),
        )
        return {col: 1.0 for col in stock_daily_returns.columns}

    market_col = aligned["__market__"].values
    var_m = float(np.var(market_col, ddof=1))

    if var_m < 1e-14:
        logger.warning("Market variance near zero — defaulting all betas to 1.0")
        return {col: 1.0 for col in stock_daily_returns.columns}

    betas: dict[str, float] = {}
    for sym in stock_daily_returns.columns:
        stock_col = aligned[sym].values
        cov_im = float(np.cov(stock_col, market_col, ddof=1)[0, 1])
        beta = cov_im / var_m
        beta = float(np.clip(beta, 0.1, 3.0))
        betas[sym] = beta
        logger.debug("Beta[%s] = %.4f", sym, beta)

    return betas


def capm_annualized_returns(
    betas: dict[str, float],
    market_daily_returns: pd.Series,
    *,
    risk_free_annual: float = DEFAULT_RISK_FREE_ANNUAL_IN,
) -> dict[str, float]:
    if market_daily_returns.empty:
        logger.warning("Empty market returns — returning rf for all CAPM estimates")
        return {sym: risk_free_annual for sym in betas}

    mu_m_daily = float(market_daily_returns.mean())
    e_rm_annual = float((1.0 + mu_m_daily) ** TRADING_DAYS_PER_YEAR - 1.0)
    market_premium = e_rm_annual - risk_free_annual

    logger.info(
        "Market annual return=%.4f  Rf=%.4f  Market premium=%.4f",
        e_rm_annual,
        risk_free_annual,
        market_premium,
    )

    capm_returns: dict[str, float] = {}
    for sym, beta in betas.items():
        er = risk_free_annual + beta * market_premium
        capm_returns[sym] = float(er)
        logger.debug("CAPM E(R)[%s] = %.4f  (beta=%.4f)", sym, er, beta)

    return capm_returns


def capm_daily_mu(
    symbols: list[str],
    market_daily_returns: pd.Series,
    *,
    risk_free_annual: float = DEFAULT_RISK_FREE_ANNUAL_IN,
    betas: dict[str, float],
) -> np.ndarray:
    capm_ann = capm_annualized_returns(
        betas,
        market_daily_returns,
        risk_free_annual=risk_free_annual,
    )
    daily_mus = []
    for sym in symbols:
        ann = capm_ann.get(sym, risk_free_annual)
        daily_equiv = float((1.0 + ann) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0)
        daily_mus.append(daily_equiv)

    return np.array(daily_mus, dtype=float)


def blend_mu(
    historical_mu_d: np.ndarray,
    capm_mu_d: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    h = np.asarray(historical_mu_d, dtype=float)
    c = np.asarray(capm_mu_d, dtype=float)
    if h.shape != c.shape:
        raise ValueError(
            f"Shape mismatch: historical {h.shape} vs capm {c.shape}"
        )
    blended = alpha * h + (1.0 - alpha) * c
    logger.info(
        "Blended mu (alpha=%.2f): hist_mean=%.6f  capm_mean=%.6f  blend_mean=%.6f",
        alpha,
        float(h.mean()),
        float(c.mean()),
        float(blended.mean()),
    )
    return blended