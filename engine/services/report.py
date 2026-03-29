"""
Assemble end-to-end portfolio analytics into a JSON-ready payload.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from models.constants import DEFAULT_RISK_FREE_ANNUAL_IN, TRADING_DAYS_PER_YEAR
from models.schemas import FullPortfolioReport
from services.frontier import random_portfolio_cloud, summarize_cloud
from services.market_data import fetch_aligned_prices, normalize_indian_tickers
from services.optimizer import (
    max_sharpe_weights,
    min_variance_weights,
    portfolio_mu_sigma_from_daily,
)
from utils.returns import annualized_return_from_daily_mean, cagr_from_price_series, daily_returns_from_prices
from utils.risk import estimate_daily_mu_cov
from utils.visualization import plot_efficient_frontier_cloud

logger = logging.getLogger(__name__)


def _weights_dict(symbols: list[str], w: np.ndarray) -> dict[str, float]:
    w = np.asarray(w, dtype=float).ravel()
    return {s: float(round(wi, 8)) for s, wi in zip(symbols, w, strict=True)}


def _equal_weights(n: int) -> np.ndarray:
    return np.ones(n, dtype=float) / n


def build_full_report(
    tickers: list[str],
    *,
    target_weights: dict[str, float] | None = None,
    start: str | None = None,
    end: str | None = None,
    risk_free_annual: float | None = None,
    random_portfolios: int = 2500,
    ridge_epsilon: float = 1e-10,
    plot_path: str | Path | None = None,
    random_seed: int = 42,
    min_history_trading_days: int = 60,
) -> dict[str, Any]:
    """
    Full pipeline: fetch → returns → mu/cov → optimize → optional plot.

    Reference portfolio for expected_return / volatility / sharpe:
    - If `target_weights` provided (sums to 1, keys match symbols): use these.
    - Else **equal weight**.

    `optimal_weights` in output = **max Sharpe** weights (explicit contract).
    """
    rf = float(DEFAULT_RISK_FREE_ANNUAL_IN if risk_free_annual is None else risk_free_annual)

    if target_weights:
        merged: dict[str, float] = {}
        for k, v in target_weights.items():
            nk = normalize_indian_tickers([k])[0]
            merged[nk] = merged.get(nk, 0.0) + float(v)
        target_weights = merged

    prices, symbols = fetch_aligned_prices(
        tickers,
        start=start,
        end=end,
        min_history_trading_days=min_history_trading_days,
    )
    rets = daily_returns_from_prices(prices, how="any")
    mu_d, cov_d, used_syms = estimate_daily_mu_cov(rets, ridge_epsilon=ridge_epsilon)
    # used_syms same as rets columns ordering from estimate_daily_mu_cov
    n = len(used_syms)

    if target_weights is None:
        w_ref = _equal_weights(n)
        ref_label = "equal_weight"
    else:
        w_ref = np.array([target_weights[s] for s in used_syms], dtype=float)
        if not np.isclose(w_ref.sum(), 1.0, atol=1e-6):
            raise ValueError("target_weights must sum to 1")
        if (w_ref < -1e-9).any():
            raise ValueError("target_weights must be non-negative (long-only)")
        w_ref = w_ref / w_ref.sum()
        ref_label = "user"

    mu_ann_ref, sig_ann_ref = portfolio_mu_sigma_from_daily(w_ref, mu_d, cov_d)
    sharpe_ref = float((mu_ann_ref - rf) / sig_ann_ref) if sig_ann_ref > 1e-12 else float("nan")
    if not np.isfinite(sharpe_ref):
        sharpe_ref = 0.0

    w_mv = min_variance_weights(cov_d)
    w_ms = max_sharpe_weights(mu_d, cov_d, rf)

    cloud = random_portfolio_cloud(mu_d, cov_d, n_samples=random_portfolios, seed=random_seed)
    frontier_stats = summarize_cloud(cloud)

    mu_mv, sig_mv = portfolio_mu_sigma_from_daily(w_mv, mu_d, cov_d)
    mu_ms, sig_ms = portfolio_mu_sigma_from_daily(w_ms, mu_d, cov_d)

    if plot_path:
        plot_efficient_frontier_cloud(
            cloud.volatilities,
            cloud.returns,
            min_var_point=(float(sig_mv), float(mu_mv)),
            max_sharpe_point=(float(sig_ms), float(mu_ms)),
            path=plot_path,
        )

    cov_list = np.asarray(cov_d, dtype=float).tolist()

    cagr_map: dict[str, float | None] = {}
    ann_mean_map: dict[str, float] = {}
    for sym in used_syms:
        cagr_v, _years = cagr_from_price_series(prices[sym])
        cagr_map[sym] = cagr_v if np.isfinite(cagr_v) else None
        md = float(rets[sym].mean())
        ann_mean_map[sym] = annualized_return_from_daily_mean(md)

    report = FullPortfolioReport(
        expected_return=float(mu_ann_ref),
        volatility=float(sig_ann_ref),
        sharpe_ratio=float(sharpe_ref),
        optimal_weights=_weights_dict(used_syms, w_ms),
        covariance_matrix=cov_list,
        symbols=list(used_syms),
        reference_portfolio=ref_label,
        min_variance_weights=_weights_dict(used_syms, w_mv),
        max_sharpe_weights=_weights_dict(used_syms, w_ms),
        risk_free_annual=rf,
        trading_days_per_year=TRADING_DAYS_PER_YEAR,
        cagr_by_symbol=cagr_map,
        annualized_mean_return_by_symbol=ann_mean_map,
        frontier_random_stats=frontier_stats,
        meta={
            "price_rows": int(len(prices)),
            "return_rows": int(len(rets)),
            "max_sharpe_annual_return": float(mu_ms),
            "max_sharpe_annual_volatility": float(sig_ms),
            "min_variance_annual_return": float(mu_mv),
            "min_variance_annual_volatility": float(sig_mv),
        },
    )
    return report.to_json_dict()
