"""JSON-serializable report schema (Pydantic v2)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FullPortfolioReport(BaseModel):
    """Stable contract for API / CLI consumers."""

    model_config = ConfigDict(extra="forbid")

    expected_return: float = Field(description="Annualized, chosen reference portfolio")
    volatility: float = Field(description="Annualized stdev of reference portfolio")
    sharpe_ratio: float
    optimal_weights: dict[str, float] = Field(
        description="Max-Sharpe long-only weights (same as max_sharpe_weights unless noted)"
    )
    covariance_matrix: list[list[float]]
    symbols: list[str]
    reference_portfolio: str = Field(default="equal_weight", description="equal_weight | user | max_sharpe")
    min_variance_weights: dict[str, float]
    max_sharpe_weights: dict[str, float]
    risk_free_annual: float
    trading_days_per_year: int
    cagr_by_symbol: dict[str, float | None] = Field(default_factory=dict)
    annualized_mean_return_by_symbol: dict[str, float] = Field(default_factory=dict)
    frontier_random_stats: dict[str, float] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)
    user_weights_raw: dict[str, float] | None = None
    user_weights_normalized: dict[str, float] | None = None
    user_expected_return: float | None = None
    user_volatility: float | None = None
    target_risk_portfolio: dict[str, float] | None = None
    target_risk_expected_return: float | None = None
    target_risk_volatility: float | None = None
    user_risk_score: float | None = None
    historical_chart_data: dict[str, Any] | None = None
    user_portfolio_cagr: float | None = None
    optimal_portfolio_cagr: float | None = None
    efficient_frontier_data: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Ensure JSON-safe (no NaN)."""
        return self.model_dump(mode="json")
