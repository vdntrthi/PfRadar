from __future__ import annotations
 
import numpy as np
import pandas as pd
import pytest
 
from utils.capm import (
    blend_mu,
    capm_annualized_returns,
    capm_daily_mu,
    compute_betas,
)
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def _make_returns(n_days: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic stock + market returns."""
    rng = np.random.default_rng(seed)
    market = pd.Series(
        rng.normal(0.0005, 0.01, n_days),
        index=pd.date_range("2020-01-01", periods=n_days, freq="B"),
        name="market",
    )
    # Stock A: beta ~1.5  (amplified market + noise)
    # Stock B: beta ~0.5  (dampened market + noise)
    stock_a = 1.5 * market.values + rng.normal(0, 0.005, n_days)
    stock_b = 0.5 * market.values + rng.normal(0, 0.005, n_days)
    stocks = pd.DataFrame(
        {"A.NS": stock_a, "B.NS": stock_b},
        index=market.index,
    )
    return stocks, market
 
 
# ---------------------------------------------------------------------------
# compute_betas
# ---------------------------------------------------------------------------
 
class TestComputeBetas:
    def test_beta_magnitudes(self):
        stocks, market = _make_returns()
        betas = compute_betas(stocks, market)
        assert set(betas.keys()) == {"A.NS", "B.NS"}
        # High-beta stock A should have beta > 1.0
        assert betas["A.NS"] > 1.0, f"Expected beta > 1.0, got {betas['A.NS']}"
        # Low-beta stock B should have beta < 1.0
        assert betas["B.NS"] < 1.0, f"Expected beta < 1.0, got {betas['B.NS']}"
 
    def test_beta_winsorized(self):
        """Artificially extreme stock should be clipped to [0.1, 3.0]."""
        rng = np.random.default_rng(99)
        n = 300
        market = pd.Series(rng.normal(0.0005, 0.01, n), index=pd.date_range("2020-01-01", periods=n, freq="B"))
        extreme = pd.DataFrame(
            {"X.NS": 10.0 * market.values},  # beta would be ~10 without winsorization
            index=market.index,
        )
        betas = compute_betas(extreme, market)
        assert betas["X.NS"] <= 3.0
 
    def test_empty_overlap_defaults_to_one(self):
        stocks, market = _make_returns(n_days=50)
        # Market with completely different dates → no overlap
        market_shifted = pd.Series(
            market.values,
            index=pd.date_range("2030-01-01", periods=len(market), freq="B"),
            name="market",
        )
        betas = compute_betas(stocks, market_shifted)
        for b in betas.values():
            assert b == 1.0
 
 
# ---------------------------------------------------------------------------
# capm_annualized_returns
# ---------------------------------------------------------------------------
 
class TestCAPMAnnualizedReturns:
    def test_beta_one_equals_market_return(self):
        """When beta=1, CAPM return should equal market return."""
        _, market = _make_returns()
        betas = {"X.NS": 1.0}
        capm = capm_annualized_returns(betas, market, risk_free_annual=0.07)
        mu_m_daily = float(market.mean())
        e_rm_ann = float((1 + mu_m_daily) ** 252 - 1)
        assert capm["X.NS"] == pytest.approx(e_rm_ann, rel=1e-4)
 
    def test_beta_zero_equals_rf(self):
        """When beta=0 (winsorized to 0.1, so use exactly 0 in formula directly)."""
        _, market = _make_returns()
        rf = 0.07
        betas = {"X.NS": 0.0}
        capm = capm_annualized_returns(betas, market, risk_free_annual=rf)
        assert capm["X.NS"] == pytest.approx(rf, rel=1e-4)
 
    def test_empty_market_returns_rf(self):
        empty_market = pd.Series(dtype=float, name="market")
        capm = capm_annualized_returns({"A.NS": 1.2}, empty_market, risk_free_annual=0.07)
        assert capm["A.NS"] == pytest.approx(0.07)
 
 
# ---------------------------------------------------------------------------
# capm_daily_mu
# ---------------------------------------------------------------------------
 
class TestCAPMDailyMu:
    def test_shape_and_order(self):
        stocks, market = _make_returns()
        betas = compute_betas(stocks, market)
        symbols = ["A.NS", "B.NS"]
        mu_d = capm_daily_mu(symbols, market, risk_free_annual=0.07, betas=betas)
        assert mu_d.shape == (2,)
 
    def test_high_beta_higher_daily_mu(self):
        """Higher beta → higher CAPM expected return (assuming positive market premium)."""
        stocks, market = _make_returns()
        betas = compute_betas(stocks, market)
        symbols = ["A.NS", "B.NS"]
        mu_d = capm_daily_mu(symbols, market, risk_free_annual=0.01, betas=betas)
        # A has higher beta, should have higher expected return
        assert mu_d[0] > mu_d[1], "High-beta stock should have higher CAPM mu"
 
 
# ---------------------------------------------------------------------------
# blend_mu
# ---------------------------------------------------------------------------
 
class TestBlendMu:
    def test_equal_blend(self):
        hist = np.array([0.001, 0.002])
        capm = np.array([0.003, 0.004])
        blended = blend_mu(hist, capm, alpha=0.5)
        expected = np.array([0.002, 0.003])
        assert np.allclose(blended, expected, atol=1e-10)
 
    def test_alpha_one_is_pure_historical(self):
        hist = np.array([0.001, 0.002])
        capm = np.array([0.999, 0.999])
        blended = blend_mu(hist, capm, alpha=1.0)
        assert np.allclose(blended, hist)
 
    def test_alpha_zero_is_pure_capm(self):
        hist = np.array([0.999, 0.999])
        capm = np.array([0.001, 0.002])
        blended = blend_mu(hist, capm, alpha=0.0)
        assert np.allclose(blended, capm)
 
    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            blend_mu(np.array([0.001]), np.array([0.001, 0.002]))
 
 
# ---------------------------------------------------------------------------
# Integration: estimate_daily_mu_cov with CAPM disabled (no network needed)
# ---------------------------------------------------------------------------
 
class TestEstimateMuCovNoCAPM:
    def test_capm_disabled_matches_original(self):
        """With use_capm_blend=False, behaviour must match original pure-historical."""
        from utils.risk import estimate_daily_mu_cov
 
        rng = np.random.default_rng(7)
        r = rng.normal(0, 0.01, size=(300, 3))
        df = pd.DataFrame(r, columns=["A.NS", "B.NS", "C.NS"])
 
        mu, cov, syms = estimate_daily_mu_cov(df, use_capm_blend=False)
        expected_mu = df.mean(axis=0).values
 
        assert np.allclose(mu, expected_mu, atol=1e-12)
        assert cov.shape == (3, 3)
        assert len(syms) == 3
 