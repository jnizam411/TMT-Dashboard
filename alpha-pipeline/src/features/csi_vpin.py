"""
CSI (Corrado-Su Implied Volatility) and VPIN Features.

These are among the strongest daily alpha factors according to
recent academic research and practitioner experience.

CSI: Implied volatility proxy from price/volume data
VPIN: Volume-Synchronized Probability of Informed Trading
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from numba import jit, prange
from scipy import stats

from ..config import get_settings

logger = structlog.get_logger(__name__)


@jit(nopython=True, cache=True)
def _bulk_volume_classification_numba(
    close: np.ndarray,
    volume: np.ndarray,
    sigma: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bulk Volume Classification (BVC) for buy/sell volume estimation.

    Uses the probability that the price change came from buying vs selling.
    """
    n = len(close)
    buy_volume = np.empty(n)
    sell_volume = np.empty(n)
    buy_volume[:] = np.nan
    sell_volume[:] = np.nan

    for i in range(1, n):
        if np.isnan(sigma[i]) or sigma[i] <= 0:
            continue

        price_change = close[i] - close[i - 1]
        z = price_change / sigma[i]

        # Probability that volume is buy vs sell
        # Using CDF of standard normal
        # Approximation of normal CDF for numba
        if z >= 0:
            buy_prob = 0.5 + 0.5 * np.sqrt(1 - np.exp(-2 * z * z / np.pi))
        else:
            buy_prob = 0.5 - 0.5 * np.sqrt(1 - np.exp(-2 * z * z / np.pi))

        buy_volume[i] = volume[i] * buy_prob
        sell_volume[i] = volume[i] * (1 - buy_prob)

    return buy_volume, sell_volume


@jit(nopython=True, cache=True)
def _compute_vpin_numba(
    buy_volume: np.ndarray,
    sell_volume: np.ndarray,
    bucket_size: int,
    n_buckets: int,
) -> np.ndarray:
    """
    Compute VPIN using volume buckets.

    VPIN = |sum(buy_vol) - sum(sell_vol)| / total_vol
    """
    n = len(buy_volume)
    vpin = np.empty(n)
    vpin[:] = np.nan

    for i in range(bucket_size * n_buckets - 1, n):
        total_buy = 0.0
        total_sell = 0.0

        for j in range(n_buckets):
            start_idx = i - (n_buckets - 1 - j) * bucket_size
            end_idx = start_idx + bucket_size

            for k in range(start_idx, min(end_idx, n)):
                if not np.isnan(buy_volume[k]):
                    total_buy += buy_volume[k]
                if not np.isnan(sell_volume[k]):
                    total_sell += sell_volume[k]

        total_vol = total_buy + total_sell
        if total_vol > 0:
            vpin[i] = abs(total_buy - total_sell) / total_vol

    return vpin


class VPINCalculator:
    """
    Volume-Synchronized Probability of Informed Trading.

    VPIN measures the toxicity of order flow and is predictive
    of future volatility and price moves.
    """

    def __init__(
        self,
        n_buckets: int = 20,
        vol_lookback: int = 20,
        settings=None,
    ):
        """
        Initialize VPIN calculator.

        Args:
            n_buckets: Number of volume buckets for VPIN
            vol_lookback: Lookback for volatility estimation
        """
        self.settings = settings or get_settings()
        self.n_buckets = n_buckets or self.settings.features.vpin_buckets
        self.vol_lookback = vol_lookback

    def compute_bvc(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute Bulk Volume Classification.

        Estimates buy and sell volume from price changes.
        """
        close = df["close"].values
        volume = df["volume"].values

        # Compute volatility for standardization
        returns = np.log(close[1:] / close[:-1])
        sigma = np.empty(len(close))
        sigma[:] = np.nan

        for i in range(self.vol_lookback, len(close)):
            sigma[i] = np.std(returns[i - self.vol_lookback:i])

        # Scale sigma to price units
        sigma = sigma * close

        buy_volume, sell_volume = _bulk_volume_classification_numba(
            close, volume, sigma
        )

        result = pd.DataFrame({
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "order_imbalance": (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-10),
        }, index=df.index)

        return result

    def compute_vpin(
        self,
        df: pd.DataFrame,
        bucket_size: int = 1,
    ) -> pd.Series:
        """
        Compute VPIN metric.

        Args:
            df: OHLCV DataFrame
            bucket_size: Size of each volume bucket in bars

        Returns:
            Series with VPIN values
        """
        bvc = self.compute_bvc(df)

        vpin = _compute_vpin_numba(
            bvc["buy_volume"].values,
            bvc["sell_volume"].values,
            bucket_size,
            self.n_buckets,
        )

        return pd.Series(vpin, index=df.index, name="vpin")

    def compute_all(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute all VPIN-related features.
        """
        features = {}

        bvc = self.compute_bvc(df)
        features["buy_volume_pct"] = bvc["buy_volume"] / df["volume"]
        features["order_imbalance"] = bvc["order_imbalance"]

        # VPIN with different bucket sizes
        for bucket_size in [1, 5]:
            vpin = self.compute_vpin(df, bucket_size=bucket_size)
            features[f"vpin_b{bucket_size}"] = vpin

        # VPIN change
        vpin = features["vpin_b1"]
        features["vpin_change_5d"] = vpin - vpin.shift(5)
        features["vpin_zscore_21d"] = (vpin - vpin.rolling(21).mean()) / vpin.rolling(21).std()

        return pd.DataFrame(features, index=df.index)


class CSICalculator:
    """
    Corrado-Su Implied Volatility Proxy.

    CSI uses skewness and kurtosis of returns to estimate
    what implied volatility should be, without option data.

    This is non-negotiable - it's currently one of the strongest
    daily alpha factors available from public data.
    """

    def __init__(
        self,
        base_window: int = 21,
        moment_windows: List[int] = [21, 63],
        settings=None,
    ):
        """
        Initialize CSI calculator.

        Args:
            base_window: Base window for volatility
            moment_windows: Windows for skewness/kurtosis
        """
        self.settings = settings or get_settings()
        self.base_window = base_window
        self.moment_windows = moment_windows

    def compute_csi(
        self,
        df: pd.DataFrame,
        window: int = 21,
    ) -> pd.DataFrame:
        """
        Compute Corrado-Su implied volatility adjustment.

        The CSI formula adjusts Black-Scholes implied vol for
        non-normality in returns.
        """
        close = df["close"]
        returns = np.log(close / close.shift(1))

        # Base volatility
        sigma = returns.rolling(window).std() * np.sqrt(252)

        # Rolling skewness
        skew = returns.rolling(window).skew()

        # Rolling excess kurtosis
        kurt = returns.rolling(window).kurt()

        # CSI adjustment factor
        # Based on Corrado-Su formula
        # Adjustment = sigma * (1 + skew/6 * d1 + (kurt-3)/24 * (d1^2 - 1))
        # For simplicity, we use d1 ≈ 0 at-the-money

        # Simplified CSI: sigma adjusted for skewness and kurtosis
        csi = sigma * (1 + skew / 6 + (kurt - 3) / 24)

        features = pd.DataFrame({
            f"csi_{window}d": csi,
            f"csi_ratio_{window}d": csi / sigma,  # CSI / base vol
            f"skew_adj_{window}d": skew / 6,
            f"kurt_adj_{window}d": (kurt - 3) / 24,
        }, index=df.index)

        return features

    def compute_vol_surface_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Features that proxy for volatility surface shape.

        Without option data, we estimate vol surface characteristics
        from realized return distribution.
        """
        features = {}

        close = df["close"]
        returns = np.log(close / close.shift(1))

        for window in self.moment_windows:
            # Volatility of volatility (convexity proxy)
            vol = returns.rolling(window).std()
            features[f"vol_of_vol_{window}d"] = vol.rolling(window).std() / vol.rolling(window).mean()

            # Skew premium proxy
            # Positive skew means calls should be expensive
            skew = returns.rolling(window).skew()
            features[f"skew_premium_{window}d"] = -skew  # Negative skew -> positive premium

            # Tail risk (kurtosis as wing proxy)
            kurt = returns.rolling(window).kurt()
            features[f"tail_risk_{window}d"] = kurt - 3

            # Upside vs downside volatility (smile asymmetry)
            up_vol = returns[returns > 0].reindex(returns.index).rolling(window).std()
            down_vol = returns[returns < 0].reindex(returns.index).rolling(window).std()
            features[f"vol_asymmetry_{window}d"] = (up_vol - down_vol) / vol

        return pd.DataFrame(features, index=df.index)

    def compute_implied_vol_change_proxy(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Proxy for implied volatility changes.

        Uses relationship between realized vol and price changes.
        """
        features = {}

        close = df["close"]
        returns = np.log(close / close.shift(1))

        # Realized vol
        rv_21 = returns.rolling(21).std() * np.sqrt(252)
        rv_5 = returns.rolling(5).std() * np.sqrt(252)

        # Vol term structure proxy (short vs long)
        features["vol_term_slope"] = rv_5 - rv_21

        # Vol mean reversion signal
        rv_252 = returns.rolling(252).std() * np.sqrt(252)
        features["vol_mean_reversion"] = (rv_21 - rv_252) / rv_252

        # Correlation of vol with price
        features["vol_price_corr"] = returns.rolling(63).corr(rv_21.diff())

        # Vol surprise (actual vs expected)
        expected_vol = rv_21.ewm(span=63).mean()
        features["vol_surprise"] = rv_21 - expected_vol

        return pd.DataFrame(features, index=df.index)

    def compute_all(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute all CSI and vol surface features.
        """
        feature_dfs = []

        for window in self.moment_windows:
            feature_dfs.append(self.compute_csi(df, window))

        feature_dfs.append(self.compute_vol_surface_features(df))
        feature_dfs.append(self.compute_implied_vol_change_proxy(df))

        result = pd.concat(feature_dfs, axis=1)

        # Remove duplicate columns
        result = result.loc[:, ~result.columns.duplicated()]

        logger.debug(f"Computed {len(result.columns)} CSI features")

        return result


class InformedTradingFeatures:
    """
    Combined informed trading detection features.

    Combines VPIN with other metrics to detect potential
    informed trading activity.
    """

    def __init__(
        self,
        settings=None,
    ):
        self.settings = settings or get_settings()
        self.vpin_calc = VPINCalculator(settings=settings)
        self.csi_calc = CSICalculator(settings=settings)

    def compute_all(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute all informed trading features.
        """
        features = {}

        # VPIN features
        vpin_features = self.vpin_calc.compute_all(df)

        # CSI features
        csi_features = self.csi_calc.compute_all(df)

        # Combine
        result = pd.concat([vpin_features, csi_features], axis=1)

        # Interaction features
        result["vpin_vol_interaction"] = result.get("vpin_b1", 0) * result.get("csi_21d", 1)

        # High VPIN + high vol = very informative
        vpin = result.get("vpin_b1", pd.Series(0, index=df.index))
        vol = result.get("csi_21d", pd.Series(1, index=df.index))
        result["informed_signal"] = (
            (vpin > vpin.rolling(63).quantile(0.8)) &
            (vol > vol.rolling(63).quantile(0.8))
        ).astype(int)

        return result
