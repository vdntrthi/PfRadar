from services.market_data import get_top_stocks, get_market_monitor
from models.schemas import TopStock, MarketMonitorSnapshot

def test_get_top_stocks():
    stocks = get_top_stocks(2)
    assert len(stocks) == 2
    assert isinstance(stocks[0], TopStock)

def test_get_market_monitor():
    monitor = get_market_monitor()
    assert isinstance(monitor, MarketMonitorSnapshot)
    assert "NIFTY 50" in monitor.indices
    assert len(monitor.top_gainers) > 0
    assert len(monitor.top_losers) > 0
