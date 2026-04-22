"""
Portfolio tracking and mutual fund logic.
"""
from __future__ import annotations
from typing import List

from models.schemas import MutualFundHolding, Portfolio
from utils.returns import calculate_simple_return

def fetch_latest_nav(scheme_name: str) -> float:
    """Mock API fetch for latest NAV."""
    mock_navs = {
        "HDFC Midcap Opportunities": 150.5,
        "SBI Small Cap": 120.2,
        "Parag Parikh Flexi Cap": 75.8
    }
    return mock_navs.get(scheme_name, 100.0)

def compute_portfolio_worth(holdings: list[MutualFundHolding]) -> Portfolio:
    """Compute total invested, current value, P&L, and returns."""
    total_invested = 0.0
    total_current = 0.0
    
    for h in holdings:
        h.invested_value = h.units * h.average_nav
        if h.current_nav is None:
            h.current_nav = fetch_latest_nav(h.scheme_name)
        h.current_value = h.units * h.current_nav
        
        total_invested += h.invested_value
        total_current += h.current_value
        
    pnl = total_current - total_invested
    ret_pct = calculate_simple_return(total_invested, total_current)
    
    return Portfolio(
        funds=holdings,
        total_invested=total_invested,
        total_current_value=total_current,
        total_profit_loss=pnl,
        overall_return_percent=ret_pct
    )
