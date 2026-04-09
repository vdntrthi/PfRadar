"""
Long-only Markowitz-style optimization using scipy.optimize (SLSQP).

- Minimum variance
- Maximum Sharpe (annualized excess return / annualized volatility)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from models.constants import DEFAULT_RISK_FREE_ANNUAL_IN, TRADING_DAYS_PER_YEAR
from models.exceptions import OptimizationFailedError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _annualized_portfolio_mean(mu_d: np.ndarray, w: np.ndarray) -> float:
    """(1 + w^T mu_d)^TDY - 1 compound annualization."""
    mu_p = float(mu_d @ w)
    td = float(TRADING_DAYS_PER_YEAR)
    return float((1.0 + mu_p) ** td - 1.0)

def _annualized_portfolio_std(cov_d: np.ndarray, w: np.ndarray) -> float:
    var_d = float(w @ cov_d @ w)
    if var_d < 0 and var_d > -1e-14:
        var_d = 0.0
    if var_d < 0:
        raise OptimizationFailedError("Negative portfolio variance")
    sigma_d = float(np.sqrt(var_d))
    return float(sigma_d * np.sqrt(float(TRADING_DAYS_PER_YEAR)))


def portfolio_mu_sigma_from_daily(
    weights: np.ndarray,
    mu_d: np.ndarray,
    cov_d: np.ndarray,
) -> tuple[float, float]:
    """Annualized expected return and volatility for weights w."""
    w = np.asarray(weights, dtype=float)
    return _annualized_portfolio_mean(mu_d, w), _annualized_portfolio_std(cov_d, w)


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    max_weight = 0.4
    w = np.clip(w, 0.0, max_weight)
    s = w.sum()
    if s <= 1e-12:
        raise OptimizationFailedError("Weights collapsed to zero")
    return w / s


def min_variance_weights(cov_d: np.ndarray, *, x0: np.ndarray | None = None) -> np.ndarray:
    """
    Minimize w^T Sigma w subject to sum w = 1, w >= 0.

    Uses a small multi-start grid: SLSQP can stagnate at the barycenter for
    ill-scaled problems; we take the best feasible objective found.
    """
    cov_d = np.asarray(cov_d, dtype=float)
    n = cov_d.shape[0]
    if cov_d.shape != (n, n):
        raise ValueError("cov_d must be square")

    def objective(w: np.ndarray) -> float:
        return float(w @ cov_d @ w)

    cons = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    max_weight = 0.4  
    bounds = tuple((0.0, max_weight) for _ in range(n))

    candidates: list[np.ndarray] = []
    if x0 is not None:
        candidates.append(np.asarray(x0, dtype=float))
    candidates.append(np.ones(n) / n)
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        candidates.append(e)

    best_x: np.ndarray | None = None
    best_obj = float("inf")
    best_msg = ""
    ok_any = False
    for w0 in candidates:
        res = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 800, "ftol": 1e-12},
        )
        x = _normalize_weights(res.x)
        obj = objective(x)
        if obj < best_obj - 1e-15:
            best_obj = obj
            best_x = x
            best_msg = res.message
            ok_any = ok_any or res.success
    if best_x is None:
        raise OptimizationFailedError("Min variance: no candidate produced weights")
    if not ok_any:
        logger.warning("min_variance: optimizer reported non-success for all inits; using best objective (%s)", best_msg)
    return best_x


def max_sharpe_weights(
    mu_d: np.ndarray,
    cov_d: np.ndarray,
    risk_free_annual: float | None = None,
    *,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """
    Maximize Sharpe = (mu_ann - rf) / sigma_ann using daily mu and covariance.
    """
    rf = float(DEFAULT_RISK_FREE_ANNUAL_IN if risk_free_annual is None else risk_free_annual)
    mu_d = np.asarray(mu_d, dtype=float)
    mu_mean = np.mean(mu_d)
    alpha = 0.6  
    mu_d = alpha * mu_d + (1 - alpha) * mu_mean
    cov_d = np.asarray(cov_d, dtype=float)
    n = len(mu_d)

    def neg_sharpe(w: np.ndarray) -> float:
        mu_ann, sig_ann = portfolio_mu_sigma_from_daily(w, mu_d, cov_d)

        if sig_ann < 1e-14:
            return 1e12

        sharpe = (mu_ann - rf) / sig_ann

        penalty_lambda = 0.15
        penalty = penalty_lambda * sig_ann

        return -float(sharpe - penalty)

    w0 = np.ones(n) / n if x0 is None else np.asarray(x0, dtype=float)
    cons = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500})
    if not res.success:
        logger.error("max_sharpe: %s", res.message)
        raise OptimizationFailedError(f"Max Sharpe failed: {res.message}")
    return _normalize_weights(res.x)
