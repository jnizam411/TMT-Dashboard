"""
Technical and Statistical Features.

Comprehensive technical indicators and rolling statistics
optimized for daily equity alpha generation.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import structlog
from numba import jit

from ..config import get_settings

logger = structlog.get_logger(__name__)


@jit(nopython=True, cache=True)
def _rolling_zscore_numba(
    x: np.ndarray,
    window: int,
) -> np.ndarray:
    """Numba-accelerated rolling z-score."""
    n = len(x)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(window - 1, n):
        window_data = x[i - window + 1:i + 1]
        mean = np.nanmean(window_data)
        std = np.nanstd(window_data)
        if std > 0:
            result[i] = (x[i] - mean) / std
        else:
            result[i] = 0.0

    return result


@jit(nopython=True, cache=True)
def _rolling_skewness_numba(
    x: np.ndarray,
    window: int,
) -> np.ndarray:
    """Numba-accelerated rolling skewness."""
    n = len(x)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(window - 1, n):
        window_data = x[i - window + 1:i + 1]

        # Remove NaN values
        valid = window_data[~np.isnan(window_data)]
        if len(valid) < 3:
            continue

        mean = np.mean(valid)
        std = np.std(valid)

        if std == 0:
            result[i] = 0.0
            continue

        m3 = np.mean((valid - mean) ** 3)
        result[i] = m3 / (std ** 3)

    return result


@jit(nopython=True, cache=True)
def _rolling_kurtosis_numba(
    x: np.ndarray,
    window: int,
) -> np.ndarray:
    """Numba-accelerated rolling kurtosis (excess)."""
    n = len(x)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(window - 1, n):
        window_data = x[i - window + 1:i + 1]

        # Remove NaN values
        valid = window_data[~np.isnan(window_data)]
        if len(valid) < 4:
            continue

        mean = np.mean(valid)
        std = np.std(valid)

        if std == 0:
            result[i] = 0.0
            continue

        m4 = np.mean((valid - mean) ** 4)
        result[i] = m4 / (std ** 4) - 3  # Excess kurtosis

    return result


class TechnicalFeatures:
    """
    Technical indicators and rolling statistics.

    All features are designed for daily frequency and
    cross-sectional equity analysis.
    """

    def __init__(
        self,
        return_windows: Optional[List[int]] = None,
        vol_windows: Optional[List[int]] = None,
        settings=None,
    ):
        self.settings = settings or get_settings()
        self.return_windows = return_windows or self.settings.features.return_windows
        self.vol_windows = vol_windows or self.settings.features.vol_windows

    def compute_returns(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
    ) -> pd.DataFrame:
        """Compute simple and log returns."""
        result = df.copy()
        prices = df[price_col]

        result["return"] = prices.pct_change()
        result["log_return"] = np.log(prices / prices.shift(1))

        return result

    def rolling_return_features(
        self,
        df: pd.DataFrame,
        return_col: str = "return",
    ) -> pd.DataFrame:
        """
        Compute rolling return-based features.

        Features:
        - Cumulative returns over various windows
        - Z-score of returns
        """
        features = {}
        returns = df[return_col]

        for window in self.return_windows:
            # Cumulative return
            features[f"cum_ret_{window}d"] = returns.rolling(window).sum()

            # Z-score of today's return
            features[f"ret_zscore_{window}d"] = pd.Series(
                _rolling_zscore_numba(returns.values, window),
                index=returns.index
            )

        return pd.DataFrame(features, index=df.index)

    def rolling_moment_features(
        self,
        df: pd.DataFrame,
        return_col: str = "return",
    ) -> pd.DataFrame:
        """
        Compute rolling statistical moments.

        Features:
        - Rolling skewness
        - Rolling kurtosis
        """
        features = {}
        returns = df[return_col].values

        for window in self.vol_windows:
            features[f"skewness_{window}d"] = pd.Series(
                _rolling_skewness_numba(returns, window),
                index=df.index
            )
            features[f"kurtosis_{window}d"] = pd.Series(
                _rolling_kurtosis_numba(returns, window),
                index=df.index
            )

        return pd.DataFrame(features, index=df.index)

    def volatility_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute volatility estimators.

        Features:
        - Close-to-close realized volatility
        - Parkinson volatility (high-low)
        - Garman-Klass volatility
        - Rogers-Satchell volatility
        - Bipower variation
        """
        features = {}

        # Ensure log prices
        log_close = np.log(df["close"])
        log_high = np.log(df["high"])
        log_low = np.log(df["low"])
        log_open = np.log(df["open"])

        # Close-to-close returns
        cc_returns = log_close.diff()

        for window in self.vol_windows:
            # Close-to-close realized vol
            features[f"realized_vol_{window}d"] = cc_returns.rolling(window).std() * np.sqrt(252)

            # Parkinson volatility
            hl_sq = (log_high - log_low) ** 2
            parkinson = np.sqrt(hl_sq.rolling(window).mean() / (4 * np.log(2))) * np.sqrt(252)
            features[f"parkinson_vol_{window}d"] = parkinson

            # Garman-Klass volatility
            gk = 0.5 * hl_sq - (2 * np.log(2) - 1) * (log_close - log_open) ** 2
            gk_vol = np.sqrt(gk.rolling(window).mean()) * np.sqrt(252)
            features[f"garman_klass_vol_{window}d"] = gk_vol

            # Rogers-Satchell volatility
            rs = (
                (log_high - log_close) * (log_high - log_open) +
                (log_low - log_close) * (log_low - log_open)
            )
            rs_vol = np.sqrt(rs.rolling(window).mean().clip(lower=0)) * np.sqrt(252)
            features[f"rogers_satchell_vol_{window}d"] = rs_vol

            # Bipower variation (jump-robust)
            abs_returns = cc_returns.abs()
            bpv = (np.pi / 2) * abs_returns * abs_returns.shift(1)
            features[f"bipower_var_{window}d"] = bpv.rolling(window - 1).mean() * 252

        return pd.DataFrame(features, index=df.index)

    def reversal_features(
        self,
        df: pd.DataFrame,
        return_col: str = "return",
    ) -> pd.DataFrame:
        """
        Short-term reversal features.

        Computes cumulative returns from day -N to day -1
        (excluding today) for reversal signal.
        """
        features = {}
        returns = df[return_col]

        # Short-term reversal: cumulative return from -5 to -1
        for lag in [5, 10, 21]:
            features[f"reversal_{lag}d"] = returns.shift(1).rolling(lag).sum()

        # Momentum vs reversal interaction
        features["momentum_5_21"] = returns.rolling(21).sum() - returns.rolling(5).sum()

        return pd.DataFrame(features, index=df.index)

    def price_level_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Price level and range features.

        Features:
        - Distance from 52-week high/low
        - Average True Range
        - Relative price position
        """
        features = {}

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # 52-week (252 days) high/low
        features["pct_from_52w_high"] = close / close.rolling(252).max() - 1
        features["pct_from_52w_low"] = close / close.rolling(252).min() - 1

        # Where in range
        range_252 = close.rolling(252).max() - close.rolling(252).min()
        features["price_position_52w"] = (close - close.rolling(252).min()) / range_252

        # Average True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features["atr_14"] = true_range.rolling(14).mean() / close
        features["atr_63"] = true_range.rolling(63).mean() / close

        return pd.DataFrame(features, index=df.index)

    def volume_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Volume-based features.

        Features:
        - Volume z-score
        - Volume relative to average
        - Price-volume trend
        """
        features = {}

        volume = df["volume"]
        close = df["close"]
        returns = close.pct_change()

        # Volume z-score
        for window in [21, 63]:
            features[f"volume_zscore_{window}d"] = pd.Series(
                _rolling_zscore_numba(volume.values, window),
                index=df.index
            )

        # Volume relative to average
        features["volume_rel_21"] = volume / volume.rolling(21).mean()
        features["volume_rel_63"] = volume / volume.rolling(63).mean()

        # Price-volume trend (return * volume)
        pv_trend = (returns * volume).rolling(21).sum()
        features["pv_trend_21"] = pv_trend / volume.rolling(21).sum()

        # Volume-weighted price
        features["vwap_deviation"] = close / (
            (close * volume).rolling(21).sum() / volume.rolling(21).sum()
        ) - 1

        return pd.DataFrame(features, index=df.index)

    def compute_all(
        self,
        df: pd.DataFrame,
        include_returns: bool = True,
    ) -> pd.DataFrame:
        """
        Compute all technical features.

        Args:
            df: OHLCV DataFrame
            include_returns: Whether to include return columns

        Returns:
            DataFrame with all features
        """
        # Ensure we have returns
        if "return" not in df.columns:
            df = self.compute_returns(df)

        feature_dfs = [
            self.rolling_return_features(df),
            self.rolling_moment_features(df),
            self.volatility_features(df),
            self.reversal_features(df),
            self.price_level_features(df),
            self.volume_features(df),
        ]

        result = pd.concat(feature_dfs, axis=1)

        if include_returns:
            result["return"] = df["return"]
            result["log_return"] = df.get("log_return", np.log(df["close"]).diff())

        return result


class MomentumFeatures:
    """
    Momentum and trend features.

    Time-series momentum factors commonly used in
    cross-sectional equity strategies.
    """

    def __init__(self):
        pass

    def compute_momentum(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        windows: List[int] = [21, 63, 126, 252],
    ) -> pd.DataFrame:
        """
        Compute momentum factors.

        Standard momentum with 1-month gap to avoid reversal.
        """
        features = {}
        prices = df[price_col]

        for window in windows:
            # Skip most recent month (21 days) to avoid reversal
            if window > 21:
                features[f"momentum_{window}d"] = (
                    prices.shift(21) / prices.shift(window) - 1
                )

        # 12-1 momentum (classic)
        features["momentum_12_1"] = prices.shift(21) / prices.shift(252) - 1

        # 6-1 momentum
        features["momentum_6_1"] = prices.shift(21) / prices.shift(126) - 1

        return pd.DataFrame(features, index=df.index)

    def compute_trend_strength(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
    ) -> pd.DataFrame:
        """
        Compute trend strength indicators.

        Features:
        - ADX (Average Directional Index)
        - Trend consistency (% of positive days)
        """
        features = {}

        close = df[price_col]
        high = df["high"]
        low = df["low"]

        # Direction consistency
        returns = close.pct_change()
        for window in [21, 63]:
            positive_pct = (returns > 0).rolling(window).mean()
            features[f"trend_consistency_{window}d"] = positive_pct - 0.5

        # Price above moving averages
        for ma_window in [21, 50, 200]:
            ma = close.rolling(ma_window).mean()
            features[f"above_ma_{ma_window}"] = (close > ma).astype(int)
            features[f"dist_from_ma_{ma_window}"] = close / ma - 1

        # Moving average crossovers
        ma_21 = close.rolling(21).mean()
        ma_63 = close.rolling(63).mean()
        features["ma_21_63_cross"] = (ma_21 > ma_63).astype(int)

        return pd.DataFrame(features, index=df.index)
