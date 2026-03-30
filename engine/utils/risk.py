"""
Risk metrics: volatility, covariance, PSD checks, optional ridge for near-singular matrices.
 
CAPM blend integrated into estimate_daily_mu_cov:
  - Fetches Nifty 50 market returns over the same date range as stock returns
  - Computes per-stock beta
  - Blends mu_d as: 0.5 * historical + 0.5 * capm  (shrinkage toward theory)
  - Falls back to pure historical if CAPM fetch fails (network unavailable)
"""
 
from __future__ import annotations
 
import logging
from datetime import date
from typing import TYPE_CHECKING
 
import numpy as np
import pandas as pd
 
from models.constants import DEFAULT_RISK_FREE_ANNUAL_IN, TRADING_DAYS_PER_YEAR
from models.exceptions import RiskModelError
 
if TYPE_CHECKING:
    pass
 
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------
 
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
 
 
# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------
 
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
 
 
# ---------------------------------------------------------------------------
# Core: estimate mu and covariance  (NOW WITH CAPM BLEND)
# ---------------------------------------------------------------------------
 
def estimate_daily_mu_cov(
    returns: pd.DataFrame,
    *,
    ridge_epsilon: float = 1e-10,
    risk_free_annual: float = DEFAULT_RISK_FREE_ANNUAL_IN,
    capm_blend_alpha: float = 0.5,
    use_capm_blend: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Column mean as mu_d, sample covariance as Sigma_d.
 
    CAPM Blend (new):
    -----------------
    When use_capm_blend=True (default), mu_d is blended:
 
        mu_d_final = alpha * mu_historical + (1 - alpha) * mu_capm_daily
 
    where alpha = capm_blend_alpha (default 0.5 → equal 50-50 split).
 
    mu_capm_daily is derived from:
        E(Ri)_annual = Rf + beta_i * (E(Rm)_annual - Rf)
        mu_capm_daily = (1 + E(Ri)_annual)^(1/252) - 1
 
    If CAPM fetch fails (e.g. no network), falls back silently to
    pure historical mu_d with a warning — no crash.
 
    Parameters
    ----------
    returns
        DataFrame of daily simple returns, columns = symbols.
    ridge_epsilon
        Ridge regularization for near-singular covariance.
    risk_free_annual
        Annual risk-free rate for CAPM (default: India 10Y G-Sec proxy 7%).
    capm_blend_alpha
        Weight on historical returns in blend. 0.5 = 50-50.
    use_capm_blend
        Set False to disable CAPM and use pure historical (useful for testing).
 
    Returns
    -------
    (mu_d, cov_d, symbols)
        mu_d  : blended daily expected returns, shape (n,)
        cov_d : daily covariance matrix, shape (n, n)  — unchanged by CAPM
        symbols: list of symbol strings
    """
    if returns.empty:
        raise RiskModelError("Empty returns frame")
 
    # --- 1. Filter usable columns ---
    cols: list[str] = []
    for c in returns.columns:
        if np.isfinite(returns[c].values).sum() >= 2:
            cols.append(str(c))
    if len(cols) < 1:
        raise RiskModelError("No usable return columns")
 
    sub = returns[cols].copy()
 
    # --- 2. Historical mu and covariance (unchanged from original) ---
    mu_hist = np.asarray(sub.mean(axis=0), dtype=float)
    cov_d = covariance_matrix_from_returns(sub)
 
    n = len(cols)
    if cov_d.shape != (n, n):
        raise RiskModelError("Internal covariance shape mismatch")
    if not np.allclose(cov_d, cov_d.T, rtol=0, atol=1e-11):
        raise RiskModelError("Covariance must be symmetric")
 
    cov_d, _ = apply_ridge_if_needed(cov_d, eps_pd=1e-8, ridge=ridge_epsilon)
    lam = smallest_eigenvalue(cov_d)
    if lam < -1e-7:
        raise RiskModelError(f"Covariance not PSD after ridge; min_eig={lam}")
 
    # --- 3. CAPM blend (new) ---
    mu_d = mu_hist  # default: pure historical
 
    if use_capm_blend:
        try:
            from utils.capm import (
                blend_mu,
                capm_daily_mu,
                compute_betas,
                fetch_market_returns,
            )
 
            # Infer date range from returns index
            idx = sub.index
            start_dt: date | None = None
            end_dt: date | None = None
            if hasattr(idx[0], "date"):
                start_dt = idx[0].date()
                end_dt = idx[-1].date()
 
            # Fetch Nifty 50 over same window
            market_rets = fetch_market_returns(start=start_dt, end=end_dt)
 
            if market_rets.empty:
                logger.warning("CAPM blend skipped — market data unavailable; using historical mu")
            else:
                # Compute betas
                betas = compute_betas(sub, market_rets)
 
                # CAPM daily mu aligned to our symbols
                mu_capm = capm_daily_mu(
                    cols,
                    market_rets,
                    risk_free_annual=risk_free_annual,
                    betas=betas,
                )
 
                # Blend: alpha * historical + (1-alpha) * capm
                mu_d = blend_mu(mu_hist, mu_capm, alpha=capm_blend_alpha)
 
                logger.info(
                    "CAPM blend applied (alpha=%.2f). "
                    "Hist mu mean=%.6f  CAPM mu mean=%.6f  Blended=%.6f",
                    capm_blend_alpha,
                    float(mu_hist.mean()),
                    float(mu_capm.mean()),
                    float(mu_d.mean()),
                )
 
        except ImportError:
            logger.warning("utils.capm not found — falling back to historical mu")
        except Exception as e:
            logger.warning("CAPM blend failed (%s) — falling back to historical mu", e)
 
    return mu_d, cov_d, cols
 