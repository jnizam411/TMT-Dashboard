"""
Fractional Differentiation for Stationary Time Series.

Implementation of fractionally differentiated features from
Marcos López de Prado's "Advances in Financial Machine Learning".

Fractional differentiation allows us to make a time series stationary
while preserving as much memory (predictive information) as possible.

Includes:
- Pure Python implementation
- Numba-accelerated implementation
- Rust extension (via PyO3) for maximum performance
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import structlog
from numba import jit, prange

from ..config import get_settings

logger = structlog.get_logger(__name__)

# Try to import Rust extension
try:
    from .fracdiff_rust import fracdiff_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    logger.debug("Rust fracdiff extension not available, using Numba fallback")


def get_weights(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Compute weights for fractional differentiation.

    The weights are derived from the binomial series expansion
    of (1 - B)^d where B is the backshift operator.

    Args:
        d: Differentiation order (0 < d < 1 for fractional)
        threshold: Minimum weight magnitude to keep

    Returns:
        Array of weights (truncated at threshold)
    """
    weights = [1.0]
    k = 1

    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1

    return np.array(weights[::-1])  # Reverse for convolution


def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Get fixed-width window fractional differentiation weights.

    FFD is more suitable for real-time applications as it uses
    a fixed lookback window.

    Args:
        d: Differentiation order
        threshold: Minimum weight magnitude

    Returns:
        Array of weights
    """
    return get_weights(d, threshold)


@jit(nopython=True, cache=True)
def _fracdiff_numba(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated fractional differentiation.

    Args:
        x: Input time series
        weights: Pre-computed weights

    Returns:
        Fractionally differentiated series
    """
    n = len(x)
    w_len = len(weights)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(w_len - 1, n):
        result[i] = 0.0
        for j in range(w_len):
            result[i] += weights[j] * x[i - w_len + 1 + j]

    return result


@jit(nopython=True, parallel=True, cache=True)
def _fracdiff_batch_numba(
    X: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Batch fractional differentiation for multiple series.

    Args:
        X: 2D array (n_samples, n_features)
        weights: Pre-computed weights

    Returns:
        2D array of fractionally differentiated series
    """
    n_samples, n_features = X.shape
    w_len = len(weights)
    result = np.empty_like(X)
    result[:] = np.nan

    for col in prange(n_features):
        for i in range(w_len - 1, n_samples):
            val = 0.0
            for j in range(w_len):
                val += weights[j] * X[i - w_len + 1 + j, col]
            result[i, col] = val

    return result


def fracdiff(
    series: Union[pd.Series, np.ndarray],
    d: float,
    threshold: float = 1e-5,
    use_rust: bool = True,
) -> Union[pd.Series, np.ndarray]:
    """
    Apply fractional differentiation to a time series.

    Args:
        series: Input time series
        d: Differentiation order (0 < d < 1 for fractional)
        threshold: Minimum weight magnitude
        use_rust: Use Rust extension if available

    Returns:
        Fractionally differentiated series (same type as input)
    """
    is_pandas = isinstance(series, pd.Series)
    if is_pandas:
        index = series.index
        x = series.values.astype(np.float64)
    else:
        x = np.asarray(series, dtype=np.float64)

    # Try Rust first if available and requested
    if use_rust and RUST_AVAILABLE:
        try:
            result = fracdiff_rust(x, d, threshold)
            if is_pandas:
                return pd.Series(result, index=index)
            return result
        except Exception as e:
            logger.warning(f"Rust fracdiff failed, falling back to Numba: {e}")

    # Numba fallback
    weights = get_weights(d, threshold)
    result = _fracdiff_numba(x, weights)

    if is_pandas:
        return pd.Series(result, index=index)
    return result


class FractionalDifferentiator:
    """
    Fractional differentiation with multiple d values.

    Computes fractionally differentiated features for various
    orders d to preserve different amounts of memory.
    """

    def __init__(
        self,
        d_values: Optional[List[float]] = None,
        threshold: float = 1e-5,
        use_rust: bool = True,
        settings=None,
    ):
        """
        Initialize fractional differentiator.

        Args:
            d_values: List of differentiation orders to compute
            threshold: Minimum weight magnitude
            use_rust: Use Rust extension if available
        """
        self.settings = settings or get_settings()
        self.d_values = d_values or self.settings.features.frac_diff_d_values
        self.threshold = threshold or self.settings.features.frac_diff_threshold
        self.use_rust = use_rust

        # Pre-compute weights for each d
        self._weights = {d: get_weights(d, threshold) for d in self.d_values}

    def transform(
        self,
        series: Union[pd.Series, np.ndarray],
        column_prefix: str = "fracdiff",
    ) -> pd.DataFrame:
        """
        Transform a series into multiple fractionally differentiated versions.

        Args:
            series: Input time series (log prices recommended)
            column_prefix: Prefix for output column names

        Returns:
            DataFrame with columns for each d value
        """
        is_pandas = isinstance(series, pd.Series)
        if is_pandas:
            index = series.index
        else:
            index = pd.RangeIndex(len(series))

        results = {}

        for d in self.d_values:
            col_name = f"{column_prefix}_d{d:.2f}".replace(".", "_")
            results[col_name] = fracdiff(
                series, d, self.threshold, self.use_rust
            )

        df = pd.DataFrame(results, index=index)
        return df

    def transform_batch(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Transform multiple columns at once.

        Args:
            df: DataFrame with columns to transform
            columns: Columns to transform (default: all numeric)

        Returns:
            DataFrame with transformed columns
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        results = []

        for col in columns:
            transformed = self.transform(df[col], column_prefix=col)
            results.append(transformed)

        return pd.concat(results, axis=1)

    def find_min_d(
        self,
        series: pd.Series,
        target_adf_pvalue: float = 0.05,
        d_range: tuple = (0.0, 1.0),
        n_steps: int = 20,
    ) -> float:
        """
        Find minimum d that achieves stationarity.

        Uses binary search to find the smallest d value
        that makes the series stationary (ADF test p-value < target).

        Args:
            series: Input time series
            target_adf_pvalue: Target p-value for ADF test
            d_range: Range of d values to search
            n_steps: Maximum number of search steps

        Returns:
            Minimum d value for stationarity
        """
        from statsmodels.tsa.stattools import adfuller

        d_low, d_high = d_range

        for _ in range(n_steps):
            d_mid = (d_low + d_high) / 2

            # Apply fractional differentiation
            diff_series = fracdiff(series.dropna(), d_mid, self.threshold)
            diff_series = diff_series.dropna()

            if len(diff_series) < 20:
                d_low = d_mid
                continue

            # ADF test
            try:
                adf_result = adfuller(diff_series, maxlag=1, regression='c')
                pvalue = adf_result[1]
            except Exception:
                d_low = d_mid
                continue

            if pvalue < target_adf_pvalue:
                d_high = d_mid
            else:
                d_low = d_mid

            if d_high - d_low < 0.01:
                break

        return (d_low + d_high) / 2


# Rust extension stub (to be compiled separately)
# This file should be in src/features/fracdiff_rust/src/lib.rs

RUST_FRACDIFF_CODE = '''
// Rust implementation for fracdiff_rust.so
// Compile with: maturin develop

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

#[pyfunction]
fn fracdiff_rust(
    py: Python,
    x: PyReadonlyArray1<f64>,
    d: f64,
    threshold: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let x = x.as_slice()?;
    let n = x.len();

    // Compute weights
    let mut weights = vec![1.0];
    let mut k = 1;
    loop {
        let w = -weights.last().unwrap() * (d - k as f64 + 1.0) / k as f64;
        if w.abs() < threshold {
            break;
        }
        weights.push(w);
        k += 1;
    }
    weights.reverse();
    let w_len = weights.len();

    // Apply fractional differentiation
    let mut result = vec![f64::NAN; n];

    for i in (w_len - 1)..n {
        let mut val = 0.0;
        for j in 0..w_len {
            val += weights[j] * x[i - w_len + 1 + j];
        }
        result[i] = val;
    }

    Ok(PyArray1::from_vec(py, result).to_owned())
}

#[pymodule]
fn fracdiff_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fracdiff_rust, m)?)?;
    Ok(())
}
'''


def create_rust_extension_template(output_dir: str = "src/features/fracdiff_rust"):
    """Create template files for Rust extension."""
    import os
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Cargo.toml
    cargo_toml = '''
[package]
name = "fracdiff_rust"
version = "0.1.0"
edition = "2021"

[lib]
name = "fracdiff_rust"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.21", features = ["extension-module"] }
numpy = "0.21"
'''

    with open(output_path / "Cargo.toml", "w") as f:
        f.write(cargo_toml)

    # src/lib.rs
    src_path = output_path / "src"
    src_path.mkdir(exist_ok=True)

    with open(src_path / "lib.rs", "w") as f:
        f.write(RUST_FRACDIFF_CODE)

    logger.info(f"Created Rust extension template in {output_path}")
