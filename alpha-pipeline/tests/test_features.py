"""Tests for feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.features.fracdiff import fracdiff, get_weights, FractionalDifferentiator
from src.features.technical import TechnicalFeatures
from src.features.microstructure import MicrostructureFeatures
from src.features.csi_vpin import VPINCalculator, CSICalculator
from src.features.pipeline import CrossSectionalProcessor


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data."""
    np.random.seed(42)
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)

    # Random walk for price
    returns = np.random.randn(n) * 0.02
    close = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_price = close * (1 + np.random.randn(n) * 0.005)

    volume = np.random.exponential(1_000_000, n)

    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "adj_close": close,
        "volume": volume,
    }, index=dates)

    return df


class TestFractionalDifferentiation:
    """Tests for fractional differentiation."""

    def test_get_weights(self):
        """Test weight computation."""
        weights = get_weights(0.5, threshold=1e-5)

        assert len(weights) > 0
        assert weights[-1] == 1.0  # Last weight (reversed) should be 1
        assert all(np.isfinite(weights))

    def test_fracdiff_basic(self, sample_ohlcv):
        """Test basic fractional differentiation."""
        series = np.log(sample_ohlcv["close"])
        result = fracdiff(series, d=0.5, use_rust=False)

        assert len(result) == len(series)
        assert result.isna().sum() > 0  # Should have some NaN at start
        assert np.isfinite(result.dropna()).all()

    def test_fracdiff_stationarity(self, sample_ohlcv):
        """Test that higher d makes series more stationary."""
        series = np.log(sample_ohlcv["close"])

        # d=1 should be stationary (returns)
        diff_1 = fracdiff(series, d=1.0, use_rust=False).dropna()

        # Check variance is lower than original
        assert diff_1.var() < series.var()

    def test_differentiator_multiple_d(self, sample_ohlcv):
        """Test differentiator with multiple d values."""
        differentiator = FractionalDifferentiator(
            d_values=[0.2, 0.4, 0.6],
            use_rust=False,
        )

        series = np.log(sample_ohlcv["close"])
        result = differentiator.transform(series)

        assert len(result.columns) == 3
        assert all("fracdiff" in c for c in result.columns)


class TestTechnicalFeatures:
    """Tests for technical features."""

    def test_compute_returns(self, sample_ohlcv):
        """Test return computation."""
        tech = TechnicalFeatures()
        result = tech.compute_returns(sample_ohlcv)

        assert "return" in result.columns
        assert "log_return" in result.columns
        assert result["return"].iloc[1:].notna().all()

    def test_rolling_features(self, sample_ohlcv):
        """Test rolling return features."""
        tech = TechnicalFeatures()
        df = tech.compute_returns(sample_ohlcv)
        result = tech.rolling_return_features(df)

        assert len(result.columns) > 0
        # Should have z-score features
        assert any("zscore" in c for c in result.columns)

    def test_volatility_features(self, sample_ohlcv):
        """Test volatility features."""
        tech = TechnicalFeatures()
        result = tech.volatility_features(sample_ohlcv)

        assert "realized_vol_63d" in result.columns
        assert "parkinson_vol_63d" in result.columns
        assert "garman_klass_vol_63d" in result.columns

    def test_compute_all(self, sample_ohlcv):
        """Test computing all features."""
        tech = TechnicalFeatures()
        result = tech.compute_all(sample_ohlcv)

        assert len(result.columns) > 20  # Should have many features


class TestMicrostructureFeatures:
    """Tests for microstructure features."""

    def test_amihud_illiquidity(self, sample_ohlcv):
        """Test Amihud illiquidity computation."""
        micro = MicrostructureFeatures()
        result = micro.amihud_illiquidity(sample_ohlcv)

        assert "amihud_21d" in result.columns
        assert "amihud_63d" in result.columns

    def test_kyle_lambda(self, sample_ohlcv):
        """Test Kyle's lambda computation."""
        micro = MicrostructureFeatures()
        result = micro.kyle_lambda(sample_ohlcv)

        assert "kyle_lambda_21d" in result.columns

    def test_spread_proxies(self, sample_ohlcv):
        """Test spread proxy estimation."""
        micro = MicrostructureFeatures()
        result = micro.spread_proxies(sample_ohlcv)

        assert "roll_spread" in result.columns
        assert "hl_spread" in result.columns


class TestCSIVPIN:
    """Tests for CSI and VPIN features."""

    def test_vpin_computation(self, sample_ohlcv):
        """Test VPIN computation."""
        vpin_calc = VPINCalculator()
        result = vpin_calc.compute_vpin(sample_ohlcv)

        assert len(result) == len(sample_ohlcv)
        # Should have some valid values
        assert result.notna().sum() > 0

    def test_csi_computation(self, sample_ohlcv):
        """Test CSI computation."""
        csi_calc = CSICalculator()
        result = csi_calc.compute_csi(sample_ohlcv, window=21)

        assert "csi_21d" in result.columns
        assert "csi_ratio_21d" in result.columns


class TestCrossSectionalProcessor:
    """Tests for cross-sectional processing."""

    def test_winsorization(self):
        """Test winsorization."""
        processor = CrossSectionalProcessor(
            winsorize_lower=0.1,
            winsorize_upper=0.9,
        )

        # Create data with outliers
        data = pd.DataFrame({
            "feature1": [1, 2, 3, 100, 5],  # 100 is outlier
            "feature2": [-50, 2, 3, 4, 5],  # -50 is outlier
        })

        result = processor.process_single_date(data)

        # Outliers should be clipped
        assert result["feature1"].max() < 100
        assert result["feature2"].min() > -50

    def test_standardization(self):
        """Test cross-sectional standardization."""
        processor = CrossSectionalProcessor()

        data = pd.DataFrame({
            "feature": [10, 20, 30, 40, 50],
        })

        result = processor.process_single_date(data)

        # Should be roughly mean 0, std ~1
        assert abs(result["feature"].mean()) < 0.1
        assert abs(result["feature"].std() - 1) < 0.5

    def test_clip_sigma(self):
        """Test clipping to N sigma."""
        processor = CrossSectionalProcessor(clip_sigma=2.0)

        data = pd.DataFrame({
            "feature": [0, 0, 0, 0, 10],  # 10 is extreme after standardization
        })

        result = processor.process_single_date(data)

        # All values should be within ±2 sigma
        assert result["feature"].abs().max() <= 2.0
