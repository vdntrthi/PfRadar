import numpy as np


def compute_beta(asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
    cov = np.cov(asset_returns, market_returns)[0, 1]
    var = np.var(market_returns)

    if var == 0:
        return 0.0

    return cov / var


def capm_expected_returns(
    returns_df,
    market_returns,
    rf_annual: float,
):
    """
    Returns expected DAILY returns using CAPM
    """

    # Convert annual → daily
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1

    # Align dates
    returns_df, market_returns = returns_df.align(
        market_returns, join="inner", axis=0
    )

    market_returns = market_returns.squeeze().values
    market_mean = np.mean(market_returns)

    # enforce realistic market premium
    market_premium = max(market_mean - rf_daily, 0.05/252)

    mu_capm = []

    for col in returns_df.columns:
        asset = returns_df[col].values

        beta = compute_beta(asset, market_returns)

        exp_return = rf_daily + beta * market_premium

        mu_capm.append(exp_return)

    return np.array(mu_capm)