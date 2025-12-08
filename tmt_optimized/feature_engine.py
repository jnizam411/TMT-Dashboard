"""
Optimized Feature Engine - High-performance feature engineering

Uses Rust core for compute-intensive operations with fallback to
NumPy/Numba for compatibility.

Optimizations:
1. Rust SIMD for FFD and rolling statistics
2. Parallel processing with Rayon
3. Memory-efficient streaming algorithms
4. Zero-copy numpy integration
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
import warnings

# Try to import Rust core
try:
    import tmt_rust_core as rust
    RUST_AVAILABLE = True
    print("[TMT] Rust acceleration: ENABLED")
except ImportError:
    RUST_AVAILABLE = False
    print("[TMT] Rust acceleration: DISABLED (using NumPy fallback)")

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    ffd_d_values: List[float] = None
    ffd_threshold: float = 1e-5
    rolling_windows: List[int] = None
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_periods: List[int] = None
    entropy_window: int = 20
    entropy_bins: int = 10
    use_rust: bool = True
    parallel: bool = True

    def __post_init__(self):
        if self.ffd_d_values is None:
            self.ffd_d_values = [0.3, 0.4, 0.5]
        if self.rolling_windows is None:
            self.rolling_windows = [5, 10, 20, 50]
        if self.bb_periods is None:
            self.bb_periods = [20, 50]


class OptimizedFeatureEngine:
    """
    High-performance feature engineering engine.

    Uses Rust core for compute-intensive operations when available,
    with NumPy/Numba fallback for compatibility.

    Usage:
        engine = OptimizedFeatureEngine()
        features = engine.engineer_all_features(df)
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize the feature engine.

        Args:
            config: Feature configuration
        """
        self.config = config or FeatureConfig()
        self.use_rust = self.config.use_rust and RUST_AVAILABLE

        if self.use_rust:
            print("[FeatureEngine] Using Rust acceleration")
        elif NUMBA_AVAILABLE:
            print("[FeatureEngine] Using Numba JIT acceleration")
        else:
            print("[FeatureEngine] Using NumPy (consider installing Rust core or Numba)")

    # ========================================================================
    # FRACTIONAL DIFFERENTIATION
    # ========================================================================

    def frac_diff_ffd(
        self,
        series: Union[np.ndarray, pd.Series],
        d: float,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Fractional differentiation using Fixed-window FFD.

        Args:
            series: Input time series
            d: Differentiation order (0 < d < 1)
            threshold: Weight cutoff threshold

        Returns:
            Fractionally differentiated series
        """
        threshold = threshold or self.config.ffd_threshold

        if isinstance(series, pd.Series):
            series = series.values

        series = np.asarray(series, dtype=np.float64)

        if self.use_rust:
            return np.array(rust.frac_diff_ffd(series, d, threshold))
        else:
            return self._frac_diff_ffd_numpy(series, d, threshold)

    def frac_diff_ffd_batch(
        self,
        series: Union[np.ndarray, pd.Series],
        d_values: Optional[List[float]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute FFD for multiple d values efficiently.

        Args:
            series: Input time series
            d_values: List of d values

        Returns:
            Dictionary mapping d values to results
        """
        d_values = d_values or self.config.ffd_d_values

        if isinstance(series, pd.Series):
            series = series.values

        series = np.asarray(series, dtype=np.float64)

        if self.use_rust:
            d_arr = np.array(d_values, dtype=np.float64)
            results = rust.frac_diff_ffd_batch(series, d_arr, self.config.ffd_threshold)
            return {f"ffd_{d}": results[i] for i, d in enumerate(d_values)}
        else:
            return {f"ffd_{d}": self._frac_diff_ffd_numpy(series, d, self.config.ffd_threshold)
                    for d in d_values}

    @staticmethod
    @jit(nopython=True, cache=True)
    def _frac_diff_ffd_numpy(series: np.ndarray, d: float, threshold: float) -> np.ndarray:
        """NumPy/Numba fallback for FFD."""
        # Compute weights
        weights = [1.0]
        k = 1
        while True:
            weight = -weights[-1] * (d - k + 1) / k
            if abs(weight) < threshold:
                break
            weights.append(weight)
            k += 1

        weights = np.array(weights)
        width = len(weights)

        if len(series) < width:
            return np.array([])

        # Reverse weights for convolution
        weights_rev = weights[::-1]

        # Apply convolution
        result = np.zeros(len(series) - width + 1)
        for i in range(len(result)):
            result[i] = np.dot(series[i:i+width], weights_rev)

        return result

    # ========================================================================
    # ROLLING STATISTICS
    # ========================================================================

    def rolling_mean(
        self,
        data: Union[np.ndarray, pd.Series],
        window: int,
    ) -> np.ndarray:
        """Rolling mean with O(n) complexity."""
        if isinstance(data, pd.Series):
            data = data.values
        data = np.asarray(data, dtype=np.float64)

        if self.use_rust:
            return np.array(rust.rolling_mean(data, window))
        else:
            return self._rolling_mean_numpy(data, window)

    def rolling_std(
        self,
        data: Union[np.ndarray, pd.Series],
        window: int,
    ) -> np.ndarray:
        """Rolling standard deviation using Welford's algorithm."""
        if isinstance(data, pd.Series):
            data = data.values
        data = np.asarray(data, dtype=np.float64)

        if self.use_rust:
            return np.array(rust.rolling_std(data, window))
        else:
            return self._rolling_std_numpy(data, window)

    def rolling_stats_batch(
        self,
        data: Union[np.ndarray, pd.Series],
        window: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute mean, std, var in single pass."""
        if isinstance(data, pd.Series):
            data = data.values
        data = np.asarray(data, dtype=np.float64)

        if self.use_rust:
            return rust.rolling_stats_batch(data, window)
        else:
            mean = self._rolling_mean_numpy(data, window)
            std = self._rolling_std_numpy(data, window)
            var = std ** 2
            return mean, std, var

    @staticmethod
    def _rolling_mean_numpy(data: np.ndarray, window: int) -> np.ndarray:
        """NumPy rolling mean."""
        result = np.full(len(data), np.nan)
        cumsum = np.cumsum(data)
        result[window-1:] = (cumsum[window-1:] - np.concatenate([[0], cumsum[:-window]])) / window
        return result

    @staticmethod
    @jit(nopython=True, cache=True)
    def _rolling_std_numpy(data: np.ndarray, window: int) -> np.ndarray:
        """Numba-optimized rolling std using Welford's algorithm."""
        n = len(data)
        result = np.full(n, np.nan)

        if n < window:
            return result

        # Initialize
        mean = 0.0
        m2 = 0.0

        for i in range(window):
            delta = data[i] - mean
            mean += delta / (i + 1)
            m2 += delta * (data[i] - mean)

        result[window - 1] = np.sqrt(m2 / (window - 1))

        # Slide window
        for i in range(window, n):
            old_val = data[i - window]
            new_val = data[i]

            # Remove old value
            old_mean = mean
            mean = (mean * window - old_val) / (window - 1)
            m2 -= (old_val - old_mean) * (old_val - mean)

            # Add new value
            delta = new_val - mean
            mean += delta / window
            m2 += delta * (new_val - mean)

            result[i] = np.sqrt(max(m2 / (window - 1), 0))

        return result

    # ========================================================================
    # TECHNICAL INDICATORS
    # ========================================================================

    def compute_rsi(
        self,
        prices: Union[np.ndarray, pd.Series],
        period: Optional[int] = None,
    ) -> np.ndarray:
        """Relative Strength Index."""
        period = period or self.config.rsi_period

        if isinstance(prices, pd.Series):
            prices = prices.values
        prices = np.asarray(prices, dtype=np.float64)

        if self.use_rust:
            return np.array(rust.compute_rsi(prices, period))
        else:
            return self._compute_rsi_numpy(prices, period)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _compute_rsi_numpy(prices: np.ndarray, period: int) -> np.ndarray:
        """Numba-optimized RSI."""
        n = len(prices)
        result = np.full(n, np.nan)

        if n < period + 1:
            return result

        # Calculate changes
        gains = np.zeros(n - 1)
        losses = np.zeros(n - 1)

        for i in range(1, n):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains[i - 1] = change
            else:
                losses[i - 1] = -change

        # Initial average
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        if avg_loss != 0:
            rs = avg_gain / avg_loss
            result[period] = 100 - (100 / (1 + rs))
        else:
            result[period] = 100

        # Smoothed RSI
        for i in range(period, n - 1):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss != 0:
                rs = avg_gain / avg_loss
                result[i + 1] = 100 - (100 / (1 + rs))
            else:
                result[i + 1] = 100

        return result

    def compute_macd(
        self,
        prices: Union[np.ndarray, pd.Series],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD indicator."""
        if isinstance(prices, pd.Series):
            prices = prices.values
        prices = np.asarray(prices, dtype=np.float64)

        if self.use_rust:
            return rust.compute_macd(
                prices,
                self.config.macd_fast,
                self.config.macd_slow,
                self.config.macd_signal,
            )
        else:
            return self._compute_macd_numpy(
                prices,
                self.config.macd_fast,
                self.config.macd_slow,
                self.config.macd_signal,
            )

    @staticmethod
    def _compute_macd_numpy(
        prices: np.ndarray,
        fast: int,
        slow: int,
        signal: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """NumPy MACD."""
        def ema(data, span):
            alpha = 2 / (span + 1)
            result = np.zeros_like(data)
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
            return result

        ema_fast = ema(prices, fast)
        ema_slow = ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def compute_bollinger_bands(
        self,
        prices: Union[np.ndarray, pd.Series],
        window: int = 20,
        num_std: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands."""
        if isinstance(prices, pd.Series):
            prices = prices.values
        prices = np.asarray(prices, dtype=np.float64)

        if self.use_rust:
            return rust.compute_bollinger_bands(prices, window, num_std)
        else:
            mean = self._rolling_mean_numpy(prices, window)
            std = self._rolling_std_numpy(prices, window)
            upper = mean + num_std * std
            lower = mean - num_std * std
            position = np.where(std > 0, (prices - mean) / std, 0)
            return upper, mean, lower, position

    # ========================================================================
    # INFORMATION THEORY
    # ========================================================================

    def rolling_entropy(
        self,
        data: Union[np.ndarray, pd.Series],
        window: Optional[int] = None,
        n_bins: Optional[int] = None,
    ) -> np.ndarray:
        """Rolling Shannon entropy."""
        window = window or self.config.entropy_window
        n_bins = n_bins or self.config.entropy_bins

        if isinstance(data, pd.Series):
            data = data.values
        data = np.asarray(data, dtype=np.float64)

        if self.use_rust:
            return np.array(rust.rolling_entropy(data, window, n_bins))
        else:
            return self._rolling_entropy_numpy(data, window, n_bins)

    @staticmethod
    def _rolling_entropy_numpy(
        data: np.ndarray,
        window: int,
        n_bins: int,
    ) -> np.ndarray:
        """NumPy rolling entropy."""
        from scipy.stats import entropy as scipy_entropy

        n = len(data)
        result = np.full(n, np.nan)

        for i in range(window - 1, n):
            window_data = data[i - window + 1:i + 1]
            window_data = window_data[~np.isnan(window_data)]

            if len(window_data) < 5:
                continue

            hist, _ = np.histogram(window_data, bins=n_bins, density=True)
            hist = hist[hist > 0]
            result[i] = scipy_entropy(hist)

        return result

    # ========================================================================
    # MICROSTRUCTURE
    # ========================================================================

    def compute_order_flow_imbalance(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> np.ndarray:
        """Order flow imbalance indicator."""
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)
        volume = np.asarray(volume, dtype=np.float64)

        if self.use_rust:
            return np.array(rust.compute_order_flow_imbalance(high, low, close, volume))
        else:
            range_hl = high - low + 1e-10
            position = (close - low) / range_hl
            normalized = 2 * position - 1
            return normalized * volume

    def compute_quote_pressure(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> np.ndarray:
        """Quote pressure indicator."""
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)

        if self.use_rust:
            return np.array(rust.compute_quote_pressure(high, low, close))
        else:
            range_hl = high - low + 1e-10
            return (2 * (close - low) / range_hl) - 1

    # ========================================================================
    # SEQUENCE CREATION
    # ========================================================================

    def create_sequences(
        self,
        data: np.ndarray,
        target: np.ndarray,
        seq_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.

        Args:
            data: Feature matrix (n_samples, n_features)
            target: Target values (n_samples,)
            seq_length: Sequence length

        Returns:
            X_sequences, y_targets
        """
        data = np.asarray(data, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)

        if self.use_rust:
            return rust.create_sequences(data, target, seq_length)
        else:
            return self._create_sequences_numpy(data, target, seq_length)

    @staticmethod
    def _create_sequences_numpy(
        data: np.ndarray,
        target: np.ndarray,
        seq_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """NumPy sequence creation using stride tricks for efficiency."""
        n_samples = len(data)
        n_features = data.shape[1] if data.ndim > 1 else 1

        if n_samples <= seq_length:
            raise ValueError("Data length must be greater than sequence length")

        n_sequences = n_samples - seq_length

        # Use stride tricks for zero-copy view where possible
        if data.flags['C_CONTIGUOUS']:
            from numpy.lib.stride_tricks import as_strided
            strides = (data.strides[0], data.strides[0], data.strides[1] if data.ndim > 1 else data.itemsize)
            shape = (n_sequences, seq_length, n_features)
            X = as_strided(data, shape=shape, strides=strides).copy()
        else:
            # Fallback for non-contiguous arrays
            X = np.zeros((n_sequences, seq_length, n_features))
            for i in range(n_sequences):
                X[i] = data[i:i + seq_length]

        y = target[seq_length:seq_length + n_sequences].copy()

        return X, y

    # ========================================================================
    # PERFORMANCE METRICS
    # ========================================================================

    def compute_ic(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> float:
        """Information Coefficient (Spearman correlation)."""
        predictions = np.asarray(predictions, dtype=np.float64)
        actuals = np.asarray(actuals, dtype=np.float64)

        if self.use_rust:
            return rust.compute_ic(predictions, actuals)
        else:
            from scipy.stats import spearmanr
            return spearmanr(predictions, actuals)[0]

    def compute_sharpe_ratio(
        self,
        returns: np.ndarray,
        periods_per_year: int = 252,
    ) -> float:
        """Annualized Sharpe ratio."""
        returns = np.asarray(returns, dtype=np.float64)

        if self.use_rust:
            return rust.compute_sharpe_ratio(returns, periods_per_year)
        else:
            mean = np.nanmean(returns)
            std = np.nanstd(returns, ddof=1)
            if std < 1e-10:
                return 0.0
            return np.sqrt(periods_per_year) * mean / std

    def compute_max_drawdown(
        self,
        returns: np.ndarray,
    ) -> float:
        """Maximum drawdown."""
        returns = np.asarray(returns, dtype=np.float64)

        if self.use_rust:
            return rust.compute_max_drawdown(returns)
        else:
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return np.min(drawdown)

    # ========================================================================
    # MASTER PIPELINE
    # ========================================================================

    def engineer_all_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'Close',
        market_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Master feature engineering pipeline.

        Args:
            df: OHLCV DataFrame
            target_col: Column to use for target
            market_df: Market benchmark for neutralization

        Returns:
            DataFrame with all engineered features
        """
        features = pd.DataFrame(index=df.index)

        # Basic returns
        returns = df[target_col].pct_change().values
        log_returns = np.log(df[target_col] / df[target_col].shift(1)).values

        # FFD features
        print("  Computing FFD features...")
        ffd_results = self.frac_diff_ffd_batch(np.log(df[target_col].values))
        for name, values in ffd_results.items():
            # Align lengths
            features[name] = np.concatenate([np.full(len(df) - len(values), np.nan), values])

        # Microstructure
        print("  Computing microstructure features...")
        features['order_flow_imbalance'] = self.compute_order_flow_imbalance(
            df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values
        )
        features['quote_pressure'] = self.compute_quote_pressure(
            df['High'].values, df['Low'].values, df['Close'].values
        )

        # Volume features
        print("  Computing volume features...")
        vol_mean = self.rolling_mean(df['Volume'].values, 20)
        features['volume_ratio'] = df['Volume'].values / (vol_mean + 1e-10)

        # Volatility
        print("  Computing volatility features...")
        features['realized_vol'] = self.rolling_std(returns, 20) * np.sqrt(252)

        # Information theory
        print("  Computing entropy features...")
        features['return_entropy'] = self.rolling_entropy(returns, 20)

        # Technical indicators
        print("  Computing technical indicators...")
        features['rsi'] = self.compute_rsi(df[target_col].values)

        macd, signal, hist = self.compute_macd(df[target_col].values)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist

        for period in self.config.bb_periods:
            _, _, _, position = self.compute_bollinger_bands(df[target_col].values, period)
            features[f'bb_position_{period}'] = position

        # Momentum
        print("  Computing momentum features...")
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = df[target_col].pct_change(period).values

        # Target
        features['target_1d'] = df[target_col].pct_change(1).shift(-1).values

        print(f"  Total features: {len(features.columns)}")

        return features
