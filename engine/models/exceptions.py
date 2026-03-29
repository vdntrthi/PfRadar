"""Domain-specific errors for data, risk, and optimization layers."""


class EngineError(Exception):
    """Base class for engine errors."""

    pass


class InvalidTickerError(EngineError):
    """User-supplied ticker cannot be normalized or is empty."""

    pass


class DataFetchError(EngineError):
    """yfinance or network returned unusable data."""

    pass


class InsufficientHistoryError(EngineError):
    """Not enough overlapping price history after alignment and cleaning."""

    pass


class RiskModelError(EngineError):
    """Covariance / PSD issues or dimension mismatch."""

    pass


class OptimizationFailedError(EngineError):
    """SciPy optimizer did not converge or produced invalid weights."""

    pass
