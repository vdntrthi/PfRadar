"""
Single source of truth for market / calendar conventions.

India-listed equities: use 252 trading days per year for annualization
(standard equity practice; aligns with NSE/BSE ~248–252 session count).
"""

TRADING_DAYS_PER_YEAR: int = 252

# India 10Y G-Sec proxy for Sharpe (configurable at runtime; document as static placeholder).
# User / ops should override from live data when available.
DEFAULT_RISK_FREE_ANNUAL_IN: float = 0.07
