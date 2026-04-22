from models.schemas import MutualFundHolding
from services.portfolio import compute_portfolio_worth

def test_compute_portfolio_worth():
    holdings = [
        MutualFundHolding(scheme_name="HDFC Midcap Opportunities", units=100, average_nav=120.0),
        MutualFundHolding(scheme_name="SBI Small Cap", units=50, average_nav=100.0)
    ]
    portfolio = compute_portfolio_worth(holdings)
    
    expected_invested = (100 * 120.0) + (50 * 100.0)
    assert portfolio.total_invested == expected_invested
    assert portfolio.total_current_value > 0
    assert isinstance(portfolio.total_profit_loss, float)
    assert isinstance(portfolio.overall_return_percent, float)
