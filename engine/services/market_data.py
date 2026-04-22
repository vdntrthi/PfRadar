"""
India-focused historical price fetch and alignment (yfinance).

Tickers are normalized to Yahoo format (NSE `.NS`, BSE `.BO`). Per-ticker history
is used for robustness across yfinance versions.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
import yfinance as yf

from models.exceptions import DataFetchError, InsufficientHistoryError, InvalidTickerError
from models.schemas import TopStock, MarketMonitorSnapshot

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_NSE_NS = ".NS"
_BSE_BO = ".BO"


def normalize_indian_tickers(raw: list[str]) -> list[str]:
    """
    Map user input to Yahoo symbols.

    - Already suffixed `.NS` / `.BO`: kept (case-normalized ticker part).
    - Bare symbols (alphanumeric & dot/hyphen): default **NSE** → append `.NS`.
    - `.NSE` alias → `.NS`.
    """
    out: list[str] = []
    for t in raw:
        s = (t or "").strip().upper()
        if not s:
            raise InvalidTickerError("Empty ticker string")
        if s.endswith(".NSE"):
            s = s.replace(".NSE", _NSE_NS)
        elif s.endswith(_NSE_NS) or s.endswith(_BSE_BO):
            if not re.match(r"^[A-Z0-9][A-Z0-9.\-&]*\.(NS|BO)$", s):
                raise InvalidTickerError(f"Malformed ticker: {t!r}")
            out.append(s)
            continue
        if not re.match(r"^[A-Z0-9][A-Z0-9.\-&]*$", s):
            raise InvalidTickerError(f"Invalid bare ticker: {t!r}")
        out.append(s + _NSE_NS)
    seen: set[str] = set()
    dedup: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup


def _parse_date(d: str | date | datetime | None) -> date | None:
    if d is None:
        return None
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return date.fromisoformat(str(d))


def fetch_adj_close_history(
    symbol: str,
    start: date | None,
    end: date | None,
) -> pd.Series:
    """Robust single-symbol adjusted close series using yf.download."""

    import time
    import pandas as pd
    import yfinance as yf

    start_s = start.isoformat() if start else None
    end_s = end.isoformat() if end else None

    last_error = None

    # Retry logic (handles Yahoo blocking / empty responses)
    for attempt in range(3):
        try:
            df = yf.download(
                symbol,
                start=start_s,
                end=end_s,
                progress=False,
                threads=False,
                auto_adjust=True,
            )

            if df is not None and not df.empty and "Close" in df.columns:
                s = df["Close"].copy()
                s.name = symbol

                if isinstance(s.index, pd.DatetimeIndex):
                    s.index = s.index.tz_localize(None).normalize()

                return s.astype(float)

        except Exception as e:
            last_error = e

        time.sleep(2)  # small delay before retry

    # If all attempts fail
    raise DataFetchError(
        f"yfinance download failed for {symbol}: {last_error or 'No data returned'}"
    )

def fetch_market_returns(start, end):
    from datetime import date, timedelta
    
    d0 = _parse_date(start) if start is not None else None
    d1 = _parse_date(end) if end is not None else None
    if d0 is None:
        d0 = (d1 or date.today()) - timedelta(days=365 * 5)
        
    df = yf.download("^NSEI", start=d0, end=d1, progress=False, auto_adjust=True)

    if df is None or df.empty:
        raise ValueError("Failed to fetch market data")

    # If yfinance returns a MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.levels[0]:
            close_series = df["Close"].iloc[:, 0]
        else:
            close_series = df.iloc[:, 0]
    else:
        close_series = df["Close"]

    if isinstance(close_series.index, pd.DatetimeIndex):
        close_series.index = close_series.index.tz_localize(None).normalize()

    returns = close_series.pct_change().dropna()
    return returns


def fetch_aligned_prices(
    tickers: list[str],
    *,
    start: str | date | None = None,
    end: str | date | None = None,
    min_history_trading_days: int = 60,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Fetch adjusted closes, inner-join dates across symbols, forward-fill sparingly.

    Parameters
    ----------
    min_history_trading_days
        Minimum number of rows required after alignment (approx. trading days).

    Raises
    ------
    InvalidTickerError, DataFetchError, InsufficientHistoryError
    """
    if min_history_trading_days < 2:
        raise ValueError("min_history_trading_days must be >= 2")

    symbols = normalize_indian_tickers(tickers)
    d0 = _parse_date(start) if start is not None else None
    d1 = _parse_date(end) if end is not None else None
    if d0 is None:
        d0 = (d1 or date.today()) - timedelta(days=365 * 5)

    series_list: list[pd.Series] = []
    for sym in symbols:
        try:
            s = fetch_adj_close_history(sym, d0, d1)
        except DataFetchError as e:
            logger.warning("Dropping %s: %s", sym, e)
            continue
        if len(s.dropna()) < min_history_trading_days:
            logger.warning(
                "Dropping %s: only %s observations (need %s)",
                sym,
                len(s.dropna()),
                min_history_trading_days,
            )
            continue
        series_list.append(s)

    if not series_list:
        raise InsufficientHistoryError("No symbols left after fetch and filtering")

    df = pd.concat(series_list, axis=1)
    df = df.sort_index()
    # Inner join: only dates all assets trade (handles holidays/staggered IPOs approximately)
    df = df.dropna(how="any")
    # Light forward-fill only for isolated gaps (optional 1 day); prefer strict inner join first
    if df.empty:
        raise InsufficientHistoryError("Aligned price frame is empty")

    if len(df) < min_history_trading_days:
        raise InsufficientHistoryError(
            f"Only {len(df)} overlapping rows; need >= {min_history_trading_days}"
        )

    used = list(df.columns)
    logger.info("Aligned prices: %s rows, symbols=%s", len(df), used)
    return df.astype(float), used

def get_top_stocks(limit: int = 5) -> list[TopStock]:
    """Fetch top-performing stocks (mock)."""
    return [
        TopStock(symbol="RELIANCE.NS", name="Reliance Ind", returns_percent=12.5, volume=1500000),
        TopStock(symbol="TCS.NS", name="TCS", returns_percent=8.2, volume=900000),
        TopStock(symbol="INFY.NS", name="Infosys", returns_percent=5.4, volume=1200000),
        TopStock(symbol="HDFCBANK.NS", name="HDFC Bank", returns_percent=4.1, volume=2100000),
        TopStock(symbol="ICICIBANK.NS", name="ICICI Bank", returns_percent=3.8, volume=1800000),
    ][:limit]

def get_market_monitor() -> MarketMonitorSnapshot:
    """Provide snapshot of market indices and sectors (mock)."""
    return MarketMonitorSnapshot(
        indices={"NIFTY 50": 22000.5, "SENSEX": 72000.1, "BANKNIFTY": 48000.2},
        top_gainers=get_top_stocks(3),
        top_losers=[
            TopStock(symbol="PAYTM.NS", name="Paytm", returns_percent=-5.5, volume=5000000),
            TopStock(symbol="WIPRO.NS", name="Wipro", returns_percent=-2.1, volume=800000)
        ],
        sector_performance={"IT": 2.5, "Banking": -1.2, "Pharma": 0.8, "FMCG": 1.1}
    )
