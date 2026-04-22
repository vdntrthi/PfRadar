"""Matplotlib plotting for efficient frontier and key portfolios."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_efficient_frontier_cloud(
    cloud_volatilities: np.ndarray,
    cloud_returns: np.ndarray,
    min_var_point: tuple[float, float],
    max_sharpe_point: tuple[float, float],
    *,
    path: str | Path,
    title: str = "Efficient frontier (random portfolios)",
) -> None:
    """
    Scatter annualized volatility (x) vs annualized return (y).

    Parameters
    ----------
    min_var_point, max_sharpe_point
        (volatility, return) for highlighted optima.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        cloud_volatilities,
        cloud_returns,
        s=8,
        c="#94a3b8",
        alpha=0.35,
        label="Random long-only",
    )
    ax.scatter(
        [min_var_point[0]],
        [min_var_point[1]],
        s=120,
        marker="*",
        c="#22c55e",
        edgecolors="black",
        zorder=5,
        label="Min variance",
    )
    ax.scatter(
        [max_sharpe_point[0]],
        [max_sharpe_point[1]],
        s=120,
        marker="D",
        c="#a855f7",
        edgecolors="black",
        zorder=5,
        label="Max Sharpe",
    )
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized expected return")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved frontier plot to %s", path)

def plot_asset_allocation(allocation: dict[str, float], path: str | Path) -> None:
    """Generate asset allocation pie chart."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = list(allocation.keys())
    sizes = list(allocation.values())
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax.axis('equal')
    ax.set_title("Asset Allocation")
    
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved asset allocation plot to %s", path)

def plot_risk_weightage(risk_weights: dict[str, float], path: str | Path) -> None:
    """Generate risk weightage pie chart."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = list(risk_weights.keys())
    sizes = list(risk_weights.values())
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
    ax.axis('equal')
    ax.set_title("Risk Weightage by Asset")
    
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved risk weightage plot to %s", path)
