"""
Monte Carlo long-only portfolios on the simplex for efficient frontier visualization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.constants import TRADING_DAYS_PER_YEAR
from services.optimizer import portfolio_mu_sigma_from_daily


@dataclass
class RandomPortfolioCloud:
    """Annualized return and volatility for each random weight vector."""

    returns: np.ndarray  # shape (n_samples,)
    volatilities: np.ndarray
    weights: np.ndarray


def random_portfolio_cloud(
    mu_d: np.ndarray,
    cov_d: np.ndarray,
    *,
    n_samples: int,
    seed: int | None = 42,
) -> RandomPortfolioCloud:
    """
    Sample weights ~ Dirichlet(1,...,1), project to simplex (already), clip & renormalize.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    mu_d = np.asarray(mu_d, dtype=float)
    n = len(mu_d)
    rng = np.random.default_rng(seed)
    # Dirichlet with alpha=1 is uniform on simplex
    w = rng.dirichlet(np.ones(n), size=n_samples)
    rets = np.empty(n_samples)
    vols = np.empty(n_samples)
    for i in range(n_samples):
        wi = w[i].astype(float)
        wi = wi / wi.sum()
        mr, sv = portfolio_mu_sigma_from_daily(wi, mu_d, cov_d)
        rets[i] = mr
        vols[i] = sv
    return RandomPortfolioCloud(
        returns=rets,
        volatilities=vols,
        weights=w,
    )


def summarize_cloud(
    cloud: RandomPortfolioCloud,
) -> dict[str, float]:
    """Basic stats for JSON embedding (numeric values only)."""
    return {
        "n_samples": float(int(len(cloud.returns))),
        "return_p50": float(np.median(cloud.returns)),
        "volatility_p50": float(np.median(cloud.volatilities)),
    }


__all__ = ["RandomPortfolioCloud", "random_portfolio_cloud", "summarize_cloud"]
