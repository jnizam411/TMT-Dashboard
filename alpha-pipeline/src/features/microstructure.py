"""
Microstructure and Liquidity Features.

Market microstructure features derived from daily OHLCV data.
These are some of the strongest alpha factors for daily equity strategies.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import structlog
from numba import jit

from ..config import get_settings

logger = structlog.get_logger(__name__)


@jit(nopython=True, cache=True)
def _rolling_regression_numba(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
) -> np.ndarray:
    """Numba-accelerated rolling OLS regression for slope (beta)."""
    n = len(x)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(window - 1, n):
        x_win = x[i - window + 1:i + 1]
        y_win = y[i - window + 1:i + 1]

        # Check for valid data
        valid_mask = ~(np.isnan(x_win) | np.isnan(y_win))
        if valid_mask.sum() < 3:
            continue

        x_valid = x_win[valid_mask]
        y_valid = y_win[valid_mask]

        x_mean = np.mean(x_valid)
        y_mean = np.mean(y_valid)

        num = np.sum((x_valid - x_mean) * (y_valid - y_mean))
        denom = np.sum((x_valid - x_mean) ** 2)

        if denom > 0:
            result[i] = num / denom

    return result


class MicrostructureFeatures:
    """
    Microstructure features from OHLCV data.

    Features:
    - Amihud illiquidity
    - Kyle's lambda
    - Order flow imbalance proxies
    - Bid-ask spread proxies
    - Volume concentration
    """

    def __init__(
        self,
        settings=None,
    ):
        self.settings = settings or get_settings()

    def amihud_illiquidity(
        self,
        df: pd.DataFrame,
        windows: List[int] = [21, 63],
    ) -> pd.DataFrame:
        """
        Amihud (2002) illiquidity measure.

        ILLIQ = |return| / dollar_volume

        Higher values indicate less liquid stocks.
        """
        features = {}

        returns = df["close"].pct_change()
        dollar_volume = df["close"] * df["volume"]

        # Daily illiquidity
        daily_illiq = returns.abs() / dollar_volume
        # Replace infinities with NaN
        daily_illiq = daily_illiq.replace([np.inf, -np.inf], np.nan)

        for window in windows:
            # Average illiquidity over window
            features[f"amihud_{window}d"] = daily_illiq.rolling(window).mean()

            # Illiquidity z-score
            illiq_mean = daily_illiq.rolling(window).mean()
            illiq_std = daily_illiq.rolling(window).std()
            features[f"amihud_zscore_{window}d"] = (daily_illiq - illiq_mean) / illiq_std

        return pd.DataFrame(features, index=df.index)

    def kyle_lambda(
        self,
        df: pd.DataFrame,
        windows: List[int] = [21, 63],
    ) -> pd.DataFrame:
        """
        Kyle's lambda - price impact measure.

        Slope of regression: return = alpha + lambda * signed_volume + epsilon

        Higher values indicate higher price impact per unit volume.
        """
        features = {}

        returns = df["close"].pct_change()
        volume = df["volume"]

        # Approximate signed volume using return direction
        signed_volume = volume * np.sign(returns)

        for window in windows:
            # Rolling regression coefficient
            lambda_values = pd.Series(
                _rolling_regression_numba(
                    signed_volume.values,
                    returns.values,
                    window
                ),
                index=df.index
            )
            features[f"kyle_lambda_{window}d"] = lambda_values

        return pd.DataFrame(features, index=df.index)

    def order_flow_imbalance(
        self,
        df: pd.DataFrame,
        windows: List[int] = [5, 21],
    ) -> pd.DataFrame:
        """
        Order flow imbalance proxies from OHLC.

        Since we don't have tick data, we infer order flow from
        price movements within the bar.
        """
        features = {}

        open_price = df["open"]
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        # Intrabar return
        intrabar_return = (close - open_price) / open_price

        # Range-based imbalance
        # Positive when close is near high, negative when near low
        range_size = high - low
        range_position = np.where(
            range_size > 0,
            (close - low) / range_size - 0.5,
            0
        )
        features["range_imbalance"] = pd.Series(range_position, index=df.index)

        # Volume-weighted imbalance
        buy_volume = volume * ((close - low) / (high - low + 1e-10))
        sell_volume = volume * ((high - close) / (high - low + 1e-10))
        features["volume_imbalance"] = (buy_volume - sell_volume) / volume

        for window in windows:
            # Cumulative imbalance
            features[f"cum_imbalance_{window}d"] = features["volume_imbalance"].rolling(window).sum()

            # Imbalance trend
            features[f"imbalance_trend_{window}d"] = features["volume_imbalance"].rolling(window).mean()

        return pd.DataFrame(features, index=df.index)

    def spread_proxies(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Bid-ask spread proxies from OHLC data.

        These are estimates since we don't have actual bid-ask data.
        """
        features = {}

        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Roll (1984) spread proxy
        # Based on covariance of consecutive price changes
        returns = close.pct_change()
        cov_returns = returns.rolling(21).cov(returns.shift(1))
        # Spread^2 = -4 * Cov(r_t, r_t-1) when Cov < 0
        roll_spread_sq = -4 * cov_returns.clip(upper=0)
        features["roll_spread"] = np.sqrt(roll_spread_sq.clip(lower=0))

        # Corwin-Schultz (2012) spread estimator
        # Based on high-low range
        beta = np.log(high / low) ** 2

        # Sum of logs for 2-day period
        h2 = high.rolling(2).max()
        l2 = low.rolling(2).min()
        gamma = np.log(h2 / l2) ** 2

        alpha_term = (np.sqrt(2 * beta.rolling(2).mean()) - np.sqrt(beta.rolling(2).mean())) / \
                     (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))

        features["cs_spread"] = 2 * (np.exp(alpha_term.clip(lower=0)) - 1) / \
                                (1 + np.exp(alpha_term.clip(lower=0)))
        features["cs_spread"] = features["cs_spread"].clip(lower=0, upper=0.1)

        # Simple high-low spread proxy
        features["hl_spread"] = (high - low) / close

        return pd.DataFrame(features, index=df.index)

    def volume_concentration(
        self,
        df: pd.DataFrame,
        windows: List[int] = [21, 63],
    ) -> pd.DataFrame:
        """
        Volume concentration measures.

        Higher concentration suggests more informed trading.
        """
        features = {}

        volume = df["volume"]
        returns = df["close"].pct_change()

        for window in windows:
            # Volume Herfindahl index (concentration)
            vol_share = volume / volume.rolling(window).sum()
            features[f"volume_hhi_{window}d"] = (vol_share ** 2).rolling(window).sum()

            # Volume on up vs down days
            up_days = returns > 0
            up_volume = volume.where(up_days, 0).rolling(window).sum()
            down_volume = volume.where(~up_days, 0).rolling(window).sum()
            features[f"up_down_volume_{window}d"] = (up_volume - down_volume) / \
                                                    (up_volume + down_volume + 1e-10)

        return pd.DataFrame(features, index=df.index)

    def volatility_of_liquidity(
        self,
        df: pd.DataFrame,
        windows: List[int] = [21, 63],
    ) -> pd.DataFrame:
        """
        Volatility of liquidity measures.

        Captures uncertainty in transaction costs.
        """
        features = {}

        volume = df["volume"]
        dollar_volume = df["close"] * volume
        returns = df["close"].pct_change()

        # Volatility of volume
        for window in windows:
            features[f"volume_vol_{window}d"] = volume.rolling(window).std() / \
                                                 volume.rolling(window).mean()

            # Volatility of Amihud
            daily_illiq = returns.abs() / dollar_volume
            daily_illiq = daily_illiq.replace([np.inf, -np.inf], np.nan)
            features[f"illiq_vol_{window}d"] = daily_illiq.rolling(window).std()

            # Autocorrelation of volume
            features[f"volume_autocorr_{window}d"] = volume.rolling(window).apply(
                lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else np.nan,
                raw=False
            )

        return pd.DataFrame(features, index=df.index)

    def compute_all(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute all microstructure features.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with all microstructure features
        """
        feature_dfs = [
            self.amihud_illiquidity(df),
            self.kyle_lambda(df),
            self.order_flow_imbalance(df),
            self.spread_proxies(df),
            self.volume_concentration(df),
            self.volatility_of_liquidity(df),
        ]

        result = pd.concat(feature_dfs, axis=1)

        # Remove any columns with all NaN
        result = result.dropna(axis=1, how="all")

        logger.debug(f"Computed {len(result.columns)} microstructure features")

        return result
