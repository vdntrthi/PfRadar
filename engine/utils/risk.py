"""
Risk metrics: volatility, covariance, PSD checks, optional ridge for near-singular matrices.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from models.constants import TRADING_DAYS_PER_YEAR
from models.exceptions import RiskModelError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def annualized_volatility_from_daily(
    daily_returns: pd.Series | np.ndarray,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Sample std of daily returns * sqrt(trading_days)."""
    x = np.asarray(daily_returns, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise RiskModelError("Need at least 2 daily returns for volatility")
    sigma_d = float(np.std(x, ddof=1))
    return float(sigma_d * np.sqrt(float(trading_days)))


def covariance_matrix_from_returns(
    returns: pd.DataFrame,
    *,
    ddof: int = 1,
) -> np.ndarray:
    """
    Sample covariance of columns (assets). Returns shape (n, n).
    """
    if returns.shape[1] < 1:
        raise RiskModelError("Returns must have at least one column")
    if returns.shape[0] < 2:
        raise RiskModelError("Need at least 2 return observations")
    mat = returns.cov(ddof=ddof)
    return np.asarray(mat.values, dtype=float)


def smallest_eigenvalue(sym: np.ndarray) -> float:
    """Minimum eigenvalue of symmetric matrix."""
    w = np.linalg.eigvalsh(sym)
    return float(np.min(w))


def validate_covariance(
    cov: np.ndarray,
    n_assets: int,
    *,
    eps_pd: float = 1e-10,
) -> None:
    """
    Check square, symmetric, and positive semi-definite (min eigenvalue >= -eps).
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise RiskModelError(f"Covariance must be square; got shape {cov.shape}")
    if cov.shape[0] != n_assets:
        raise RiskModelError(f"Cov dim {cov.shape[0]} != n_assets {n_assets}")
    if not np.allclose(cov, cov.T, rtol=0, atol=1e-12):
        raise RiskModelError("Covariance must be symmetric")
    lam_min = smallest_eigenvalue(cov)
    if lam_min < -eps_pd:
        raise RiskModelError(f"Covariance not PSD within tolerance; min_eig={lam_min}")


def apply_ridge_if_needed(
    cov: np.ndarray,
    *,
    eps_pd: float = 1e-10,
    ridge: float = 1e-10,
) -> tuple[np.ndarray, bool]:
    """
    If min eigenvalue < eps_pd, add `ridge * I` and log.

    Returns
    -------
    (cov_out_maybe_ridge, ridge_applied)
    """
    cov = np.asarray(cov, dtype=float).copy()
    lam_min = smallest_eigenvalue(cov)
    if lam_min >= eps_pd:
        return cov, False
    logger.warning(
        "Near-singular covariance (min_eig=%.2e); applying ridge=%.2e",
        lam_min,
        ridge,
    )
    n = cov.shape[0]
    cov = cov + np.eye(n, dtype=float) * ridge
    return cov, True


def estimate_daily_mu_cov(
    returns: pd.DataFrame,
    *,
    ridge_epsilon: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Column mean as mu_d, sample covariance as Sigma_d; drop non-finite columns first.

    Returns
    -------
    (mu_d, cov_d, symbols)
    """
    if returns.empty:
        raise RiskModelError("Empty returns frame")
    # Only use columns with finite data
    cols: list[str] = []
    for c in returns.columns:
        if np.isfinite(returns[c].values).sum() >= 2:
            cols.append(str(c))
    if len(cols) < 1:
        raise RiskModelError("No usable return columns")
    sub = returns[cols].copy()
    mu_d = np.asarray(sub.mean(axis=0), dtype=float)
    cov_d = covariance_matrix_from_returns(sub)
    n = len(cols)
    if cov_d.shape != (n, n):
        raise RiskModelError("Internal covariance shape mismatch")
    if not np.allclose(cov_d, cov_d.T, rtol=0, atol=1e-11):
        raise RiskModelError("Covariance must be symmetric")
    # Sample covariance can be nearly singular; ridge before strict PSD check
    cov_d, _ = apply_ridge_if_needed(cov_d, eps_pd=1e-8, ridge=ridge_epsilon)
    lam = smallest_eigenvalue(cov_d)
    if lam < -1e-7:
        raise RiskModelError(f"Covariance not PSD after ridge; min_eig={lam}")
    return mu_d, cov_d, cols

def compute_risk_weightage(asset_weights: dict[str, float], volatilities: dict[str, float]) -> dict[str, float]:
    """Compute risk contribution per asset (mocked as weight * volatility for simplicity)."""
    risk_contributions = {}
    total_risk = 0.0
    for asset, weight in asset_weights.items():
        vol = volatilities.get(asset, 0.1)
        contrib = weight * vol
        risk_contributions[asset] = contrib
        total_risk += contrib
        
    if total_risk == 0:
        return {k: 0.0 for k in asset_weights}
        
    return {k: (v / total_risk) * 100 for k, v in risk_contributions.items()}
