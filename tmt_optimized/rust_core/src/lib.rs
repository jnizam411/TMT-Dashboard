//! TMT Rust Core - High-performance computations for quantitative finance
//!
//! Optimized for Apple M-series (ARM64) with:
//! - SIMD vectorization via ARM NEON
//! - Parallel processing via Rayon
//! - Zero-copy numpy integration
//! - Memory-efficient streaming algorithms

use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;
use ndarray::{Array1, Array2, Array3, Axis, s};
use std::f64;

// ============================================================================
// MODULE 1: FRACTIONAL DIFFERENTIATION (FFD)
// ============================================================================
//
// The FFD operation is a convolution that achieves stationarity while
// preserving memory. This is the most compute-intensive operation in
// the feature engineering pipeline.
//
// Optimization strategy:
// 1. Pre-compute weights once
// 2. Use SIMD for dot products (automatic on ARM64)
// 3. Parallelize across time steps with Rayon
// ============================================================================

/// Compute FFD weights using the recursive formula
/// w_k = -w_{k-1} * (d - k + 1) / k
#[inline]
fn compute_ffd_weights(d: f64, threshold: f64) -> Vec<f64> {
    let mut weights = vec![1.0];
    let mut k = 1;

    loop {
        let weight = -weights[k - 1] * (d - k as f64 + 1.0) / k as f64;
        if weight.abs() < threshold {
            break;
        }
        weights.push(weight);
        k += 1;
    }

    weights
}

/// SIMD-optimized dot product for ARM NEON
/// Uses automatic vectorization on aarch64
#[inline]
fn simd_dot_product(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());

    // Rust compiler auto-vectorizes this for ARM NEON
    // Using chunks for better cache utilization
    let mut sum = 0.0;
    let chunks = a.len() / 4;

    for i in 0..chunks {
        let idx = i * 4;
        sum += a[idx] * b[idx]
            + a[idx + 1] * b[idx + 1]
            + a[idx + 2] * b[idx + 2]
            + a[idx + 3] * b[idx + 3];
    }

    // Handle remainder
    for i in (chunks * 4)..a.len() {
        sum += a[i] * b[i];
    }

    sum
}

/// Parallel FFD computation using Rayon
///
/// This is the main workhorse - parallelizes the convolution across
/// all time steps, with each thread computing one output value.
fn frac_diff_ffd_parallel(series: &[f64], d: f64, threshold: f64) -> Vec<f64> {
    let weights = compute_ffd_weights(d, threshold);
    let width = weights.len();

    if series.len() < width {
        return vec![];
    }

    // Reverse weights once for convolution
    let weights_rev: Vec<f64> = weights.iter().rev().cloned().collect();

    // Parallel computation using Rayon
    let result: Vec<f64> = (0..=(series.len() - width))
        .into_par_iter()
        .map(|i| {
            simd_dot_product(&series[i..i + width], &weights_rev)
        })
        .collect();

    result
}

/// Python-exposed FFD function
#[pyfunction]
fn frac_diff_ffd(
    py: Python<'_>,
    series: PyReadonlyArray1<f64>,
    d: f64,
    threshold: Option<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let threshold = threshold.unwrap_or(1e-5);
    let series_slice = series.as_slice()?;

    let result = frac_diff_ffd_parallel(series_slice, d, threshold);

    Ok(Array1::from_vec(result).into_pyarray(py).into())
}

/// Batch FFD for multiple d values (common use case)
#[pyfunction]
fn frac_diff_ffd_batch(
    py: Python<'_>,
    series: PyReadonlyArray1<f64>,
    d_values: PyReadonlyArray1<f64>,
    threshold: Option<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let threshold = threshold.unwrap_or(1e-5);
    let series_slice = series.as_slice()?;
    let d_slice = d_values.as_slice()?;

    // Parallel across d values
    let results: Vec<Vec<f64>> = d_slice
        .par_iter()
        .map(|&d| frac_diff_ffd_parallel(series_slice, d, threshold))
        .collect();

    // Find minimum length (different d values may produce different lengths)
    let min_len = results.iter().map(|v| v.len()).min().unwrap_or(0);

    // Create 2D array
    let mut arr = Array2::zeros((d_slice.len(), min_len));
    for (i, result) in results.iter().enumerate() {
        for (j, &val) in result.iter().take(min_len).enumerate() {
            arr[[i, j]] = val;
        }
    }

    Ok(arr.into_pyarray(py).into())
}

// ============================================================================
// MODULE 2: ROLLING STATISTICS (Welford's Algorithm)
// ============================================================================
//
// Uses online algorithms for O(n) complexity instead of O(n*window).
// Memory-efficient streaming computation perfect for limited RAM.
// ============================================================================

/// Welford's online algorithm state
struct WelfordState {
    n: usize,
    mean: f64,
    m2: f64,
}

impl WelfordState {
    fn new() -> Self {
        WelfordState { n: 0, mean: 0.0, m2: 0.0 }
    }

    #[inline]
    fn update(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    #[inline]
    fn remove(&mut self, x: f64) {
        if self.n == 0 { return; }
        if self.n == 1 {
            self.n = 0;
            self.mean = 0.0;
            self.m2 = 0.0;
            return;
        }
        let delta = x - self.mean;
        self.mean = (self.mean * self.n as f64 - x) / (self.n - 1) as f64;
        let delta2 = x - self.mean;
        self.m2 -= delta * delta2;
        self.n -= 1;
    }

    #[inline]
    fn variance(&self) -> f64 {
        if self.n < 2 { return f64::NAN; }
        self.m2 / (self.n - 1) as f64
    }

    #[inline]
    fn std(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Rolling mean using streaming algorithm
#[pyfunction]
fn rolling_mean(
    py: Python<'_>,
    data: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let data_slice = data.as_slice()?;
    let n = data_slice.len();

    if n < window {
        return Ok(Array1::from_vec(vec![f64::NAN; n]).into_pyarray(py).into());
    }

    let mut result = vec![f64::NAN; n];
    let mut sum = 0.0;

    // Initialize window
    for i in 0..window {
        sum += data_slice[i];
    }
    result[window - 1] = sum / window as f64;

    // Slide window
    for i in window..n {
        sum += data_slice[i] - data_slice[i - window];
        result[i] = sum / window as f64;
    }

    Ok(Array1::from_vec(result).into_pyarray(py).into())
}

/// Rolling standard deviation using Welford's algorithm
#[pyfunction]
fn rolling_std(
    py: Python<'_>,
    data: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let data_slice = data.as_slice()?;
    let n = data_slice.len();

    if n < window {
        return Ok(Array1::from_vec(vec![f64::NAN; n]).into_pyarray(py).into());
    }

    let mut result = vec![f64::NAN; n];
    let mut state = WelfordState::new();

    // Initialize window
    for i in 0..window {
        state.update(data_slice[i]);
    }
    result[window - 1] = state.std();

    // Slide window using add/remove
    for i in window..n {
        state.remove(data_slice[i - window]);
        state.update(data_slice[i]);
        result[i] = state.std();
    }

    Ok(Array1::from_vec(result).into_pyarray(py).into())
}

/// Rolling variance
#[pyfunction]
fn rolling_var(
    py: Python<'_>,
    data: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let data_slice = data.as_slice()?;
    let n = data_slice.len();

    if n < window {
        return Ok(Array1::from_vec(vec![f64::NAN; n]).into_pyarray(py).into());
    }

    let mut result = vec![f64::NAN; n];
    let mut state = WelfordState::new();

    for i in 0..window {
        state.update(data_slice[i]);
    }
    result[window - 1] = state.variance();

    for i in window..n {
        state.remove(data_slice[i - window]);
        state.update(data_slice[i]);
        result[i] = state.variance();
    }

    Ok(Array1::from_vec(result).into_pyarray(py).into())
}

/// Compute multiple rolling statistics in one pass (more efficient)
#[pyfunction]
fn rolling_stats_batch(
    py: Python<'_>,
    data: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let data_slice = data.as_slice()?;
    let n = data_slice.len();

    let nan_array = || Array1::from_vec(vec![f64::NAN; n]);

    if n < window {
        return Ok((
            nan_array().into_pyarray(py).into(),
            nan_array().into_pyarray(py).into(),
            nan_array().into_pyarray(py).into(),
        ));
    }

    let mut means = vec![f64::NAN; n];
    let mut stds = vec![f64::NAN; n];
    let mut vars = vec![f64::NAN; n];

    let mut state = WelfordState::new();

    for i in 0..window {
        state.update(data_slice[i]);
    }
    means[window - 1] = state.mean;
    vars[window - 1] = state.variance();
    stds[window - 1] = state.std();

    for i in window..n {
        state.remove(data_slice[i - window]);
        state.update(data_slice[i]);
        means[i] = state.mean;
        vars[i] = state.variance();
        stds[i] = state.std();
    }

    Ok((
        Array1::from_vec(means).into_pyarray(py).into(),
        Array1::from_vec(stds).into_pyarray(py).into(),
        Array1::from_vec(vars).into_pyarray(py).into(),
    ))
}

// ============================================================================
// MODULE 3: SEQUENCE CREATION (Zero-Copy Optimized)
// ============================================================================
//
// Creates sliding window sequences for LSTM input.
// Uses pre-allocation and parallel copying for efficiency.
// ============================================================================

/// Create sequences for LSTM training
/// Returns (X_sequences, y_targets)
#[pyfunction]
fn create_sequences(
    py: Python<'_>,
    data: PyReadonlyArray2<f64>,
    target: PyReadonlyArray1<f64>,
    seq_length: usize,
) -> PyResult<(Py<PyArray3<f64>>, Py<PyArray1<f64>>)> {
    let data_arr = data.as_array();
    let target_slice = target.as_slice()?;

    let n_samples = data_arr.shape()[0];
    let n_features = data_arr.shape()[1];

    if n_samples <= seq_length {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Data length must be greater than sequence length"
        ));
    }

    let n_sequences = n_samples - seq_length;

    // Pre-allocate output arrays
    let mut x_data = Array3::<f64>::zeros((n_sequences, seq_length, n_features));
    let mut y_data = Array1::<f64>::zeros(n_sequences);

    // Parallel sequence creation using Rayon
    x_data
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut seq)| {
            for j in 0..seq_length {
                for k in 0..n_features {
                    seq[[j, k]] = data_arr[[i + j, k]];
                }
            }
        });

    // Copy targets (sequential as it's small)
    for i in 0..n_sequences {
        y_data[i] = target_slice[i + seq_length];
    }

    Ok((
        x_data.into_pyarray(py).into(),
        y_data.into_pyarray(py).into(),
    ))
}

/// Create sequences with strided view hint
/// Returns indices for creating strided tensors in PyTorch
#[pyfunction]
fn create_sequence_indices(
    n_samples: usize,
    seq_length: usize,
) -> PyResult<(Vec<usize>, Vec<usize>)> {
    if n_samples <= seq_length {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Data length must be greater than sequence length"
        ));
    }

    let n_sequences = n_samples - seq_length;
    let start_indices: Vec<usize> = (0..n_sequences).collect();
    let target_indices: Vec<usize> = (seq_length..n_samples).collect();

    Ok((start_indices, target_indices))
}

// ============================================================================
// MODULE 4: TECHNICAL INDICATORS (Vectorized)
// ============================================================================
//
// Optimized implementations of common technical indicators.
// Uses SIMD where beneficial and streaming where memory-efficient.
// ============================================================================

/// RSI (Relative Strength Index) - streaming implementation
#[pyfunction]
fn compute_rsi(
    py: Python<'_>,
    prices: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let prices_slice = prices.as_slice()?;
    let n = prices_slice.len();

    if n < period + 1 {
        return Ok(Array1::from_vec(vec![f64::NAN; n]).into_pyarray(py).into());
    }

    let mut result = vec![f64::NAN; n];
    let mut gains = Vec::with_capacity(n - 1);
    let mut losses = Vec::with_capacity(n - 1);

    // Calculate price changes
    for i in 1..n {
        let change = prices_slice[i] - prices_slice[i - 1];
        gains.push(if change > 0.0 { change } else { 0.0 });
        losses.push(if change < 0.0 { -change } else { 0.0 });
    }

    // Initial average
    let mut avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;

    // First RSI value
    let rs = if avg_loss != 0.0 { avg_gain / avg_loss } else { f64::INFINITY };
    result[period] = 100.0 - (100.0 / (1.0 + rs));

    // Smoothed RSI
    for i in period..gains.len() {
        avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

        let rs = if avg_loss != 0.0 { avg_gain / avg_loss } else { f64::INFINITY };
        result[i + 1] = 100.0 - (100.0 / (1.0 + rs));
    }

    Ok(Array1::from_vec(result).into_pyarray(py).into())
}

/// EMA (Exponential Moving Average)
#[pyfunction]
fn compute_ema(
    py: Python<'_>,
    data: PyReadonlyArray1<f64>,
    span: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let data_slice = data.as_slice()?;
    let n = data_slice.len();

    if n == 0 {
        return Ok(Array1::from_vec(vec![]).into_pyarray(py).into());
    }

    let alpha = 2.0 / (span as f64 + 1.0);
    let mut result = vec![0.0; n];
    result[0] = data_slice[0];

    for i in 1..n {
        result[i] = alpha * data_slice[i] + (1.0 - alpha) * result[i - 1];
    }

    Ok(Array1::from_vec(result).into_pyarray(py).into())
}

/// MACD (Moving Average Convergence Divergence)
/// Returns (macd_line, signal_line, histogram)
#[pyfunction]
fn compute_macd(
    py: Python<'_>,
    prices: PyReadonlyArray1<f64>,
    fast_period: Option<usize>,
    slow_period: Option<usize>,
    signal_period: Option<usize>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let fast = fast_period.unwrap_or(12);
    let slow = slow_period.unwrap_or(26);
    let signal = signal_period.unwrap_or(9);

    let prices_slice = prices.as_slice()?;
    let n = prices_slice.len();

    if n == 0 {
        let empty = Array1::from_vec(vec![]);
        return Ok((
            empty.clone().into_pyarray(py).into(),
            empty.clone().into_pyarray(py).into(),
            empty.into_pyarray(py).into(),
        ));
    }

    // Compute EMAs
    let alpha_fast = 2.0 / (fast as f64 + 1.0);
    let alpha_slow = 2.0 / (slow as f64 + 1.0);
    let alpha_signal = 2.0 / (signal as f64 + 1.0);

    let mut ema_fast = vec![0.0; n];
    let mut ema_slow = vec![0.0; n];
    let mut macd_line = vec![0.0; n];
    let mut signal_line = vec![0.0; n];
    let mut histogram = vec![0.0; n];

    ema_fast[0] = prices_slice[0];
    ema_slow[0] = prices_slice[0];

    for i in 1..n {
        ema_fast[i] = alpha_fast * prices_slice[i] + (1.0 - alpha_fast) * ema_fast[i - 1];
        ema_slow[i] = alpha_slow * prices_slice[i] + (1.0 - alpha_slow) * ema_slow[i - 1];
        macd_line[i] = ema_fast[i] - ema_slow[i];
    }

    signal_line[0] = macd_line[0];
    for i in 1..n {
        signal_line[i] = alpha_signal * macd_line[i] + (1.0 - alpha_signal) * signal_line[i - 1];
        histogram[i] = macd_line[i] - signal_line[i];
    }

    Ok((
        Array1::from_vec(macd_line).into_pyarray(py).into(),
        Array1::from_vec(signal_line).into_pyarray(py).into(),
        Array1::from_vec(histogram).into_pyarray(py).into(),
    ))
}

/// Bollinger Bands
/// Returns (upper_band, middle_band, lower_band, bb_position)
#[pyfunction]
fn compute_bollinger_bands(
    py: Python<'_>,
    prices: PyReadonlyArray1<f64>,
    window: usize,
    num_std: Option<f64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let num_std = num_std.unwrap_or(2.0);
    let prices_slice = prices.as_slice()?;
    let n = prices_slice.len();

    let mut upper = vec![f64::NAN; n];
    let mut middle = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];
    let mut position = vec![f64::NAN; n];

    if n < window {
        return Ok((
            Array1::from_vec(upper).into_pyarray(py).into(),
            Array1::from_vec(middle).into_pyarray(py).into(),
            Array1::from_vec(lower).into_pyarray(py).into(),
            Array1::from_vec(position).into_pyarray(py).into(),
        ));
    }

    let mut state = WelfordState::new();

    for i in 0..window {
        state.update(prices_slice[i]);
    }

    let std_val = state.std();
    middle[window - 1] = state.mean;
    upper[window - 1] = state.mean + num_std * std_val;
    lower[window - 1] = state.mean - num_std * std_val;
    position[window - 1] = if std_val > 0.0 {
        (prices_slice[window - 1] - state.mean) / std_val
    } else {
        0.0
    };

    for i in window..n {
        state.remove(prices_slice[i - window]);
        state.update(prices_slice[i]);

        let std_val = state.std();
        middle[i] = state.mean;
        upper[i] = state.mean + num_std * std_val;
        lower[i] = state.mean - num_std * std_val;
        position[i] = if std_val > 0.0 {
            (prices_slice[i] - state.mean) / std_val
        } else {
            0.0
        };
    }

    Ok((
        Array1::from_vec(upper).into_pyarray(py).into(),
        Array1::from_vec(middle).into_pyarray(py).into(),
        Array1::from_vec(lower).into_pyarray(py).into(),
        Array1::from_vec(position).into_pyarray(py).into(),
    ))
}

// ============================================================================
// MODULE 5: INFORMATION THEORY
// ============================================================================

/// Rolling Shannon entropy
#[pyfunction]
fn rolling_entropy(
    py: Python<'_>,
    data: PyReadonlyArray1<f64>,
    window: usize,
    n_bins: Option<usize>,
) -> PyResult<Py<PyArray1<f64>>> {
    let n_bins = n_bins.unwrap_or(10);
    let data_slice = data.as_slice()?;
    let n = data_slice.len();

    if n < window {
        return Ok(Array1::from_vec(vec![f64::NAN; n]).into_pyarray(py).into());
    }

    let mut result = vec![f64::NAN; n];

    // Compute entropy for each window
    for i in (window - 1)..n {
        let window_data: Vec<f64> = data_slice[(i + 1 - window)..=i]
            .iter()
            .filter(|x| !x.is_nan())
            .cloned()
            .collect();

        if window_data.len() < 5 {
            continue;
        }

        // Find min/max for histogram
        let min_val = window_data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = window_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < 1e-10 {
            result[i] = 0.0;
            continue;
        }

        // Create histogram
        let bin_width = (max_val - min_val) / n_bins as f64;
        let mut hist = vec![0usize; n_bins];

        for &val in &window_data {
            let bin = ((val - min_val) / bin_width).floor() as usize;
            let bin = bin.min(n_bins - 1);
            hist[bin] += 1;
        }

        // Compute entropy
        let total = window_data.len() as f64;
        let entropy: f64 = hist.iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f64 / total;
                -p * p.ln()
            })
            .sum();

        result[i] = entropy;
    }

    Ok(Array1::from_vec(result).into_pyarray(py).into())
}

// ============================================================================
// MODULE 6: MARKET MICROSTRUCTURE
// ============================================================================

/// Order Flow Imbalance
#[pyfunction]
fn compute_order_flow_imbalance(
    py: Python<'_>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;

    let n = high_slice.len();

    let result: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let range = high_slice[i] - low_slice[i] + 1e-10;
            let price_position = (close_slice[i] - low_slice[i]) / range;
            let normalized = 2.0 * price_position - 1.0;
            normalized * volume_slice[i]
        })
        .collect();

    Ok(Array1::from_vec(result).into_pyarray(py).into())
}

/// Quote Pressure
#[pyfunction]
fn compute_quote_pressure(
    py: Python<'_>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;

    let n = high_slice.len();

    let result: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let range = high_slice[i] - low_slice[i] + 1e-10;
            (2.0 * (close_slice[i] - low_slice[i]) / range) - 1.0
        })
        .collect();

    Ok(Array1::from_vec(result).into_pyarray(py).into())
}

// ============================================================================
// MODULE 7: PERFORMANCE METRICS
// ============================================================================

/// Information Coefficient (Spearman correlation)
#[pyfunction]
fn compute_ic(
    predictions: PyReadonlyArray1<f64>,
    actuals: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let pred_slice = predictions.as_slice()?;
    let actual_slice = actuals.as_slice()?;

    if pred_slice.len() != actual_slice.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Arrays must have same length"
        ));
    }

    let n = pred_slice.len();
    if n < 2 {
        return Ok(f64::NAN);
    }

    // Compute ranks
    fn rank(data: &[f64]) -> Vec<f64> {
        let mut indexed: Vec<(usize, f64)> = data.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut ranks = vec![0.0; data.len()];
        for (rank, (idx, _)) in indexed.iter().enumerate() {
            ranks[*idx] = rank as f64 + 1.0;
        }
        ranks
    }

    let pred_ranks = rank(pred_slice);
    let actual_ranks = rank(actual_slice);

    // Compute Spearman correlation
    let mean_pred: f64 = pred_ranks.iter().sum::<f64>() / n as f64;
    let mean_actual: f64 = actual_ranks.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_pred = 0.0;
    let mut var_actual = 0.0;

    for i in 0..n {
        let dp = pred_ranks[i] - mean_pred;
        let da = actual_ranks[i] - mean_actual;
        cov += dp * da;
        var_pred += dp * dp;
        var_actual += da * da;
    }

    let denom = (var_pred * var_actual).sqrt();
    if denom < 1e-10 {
        return Ok(0.0);
    }

    Ok(cov / denom)
}

/// Sharpe Ratio
#[pyfunction]
fn compute_sharpe_ratio(
    returns: PyReadonlyArray1<f64>,
    periods_per_year: Option<usize>,
) -> PyResult<f64> {
    let periods = periods_per_year.unwrap_or(252);
    let returns_slice = returns.as_slice()?;

    let n = returns_slice.len();
    if n < 2 {
        return Ok(f64::NAN);
    }

    let mean: f64 = returns_slice.iter().sum::<f64>() / n as f64;
    let variance: f64 = returns_slice.iter()
        .map(|&r| (r - mean).powi(2))
        .sum::<f64>() / (n - 1) as f64;
    let std = variance.sqrt();

    if std < 1e-10 {
        return Ok(0.0);
    }

    Ok((periods as f64).sqrt() * mean / std)
}

/// Maximum Drawdown
#[pyfunction]
fn compute_max_drawdown(
    returns: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let returns_slice = returns.as_slice()?;

    if returns_slice.is_empty() {
        return Ok(0.0);
    }

    let mut cumulative = 1.0;
    let mut peak = 1.0;
    let mut max_dd = 0.0;

    for &ret in returns_slice {
        cumulative *= 1.0 + ret;
        peak = peak.max(cumulative);
        let dd = (cumulative - peak) / peak;
        max_dd = max_dd.min(dd);
    }

    Ok(max_dd)
}

// ============================================================================
// MODULE 8: MEMORY UTILITIES
// ============================================================================

/// Get optimal batch size based on available memory
#[pyfunction]
fn suggest_batch_size(
    n_samples: usize,
    seq_length: usize,
    n_features: usize,
    available_memory_gb: f64,
    dtype_bytes: Option<usize>,
) -> PyResult<usize> {
    let dtype_size = dtype_bytes.unwrap_or(4); // float32

    // Memory per sample: seq_length * n_features * dtype_size
    let bytes_per_sample = seq_length * n_features * dtype_size;

    // Reserve 50% of memory for model and gradients
    let usable_memory = (available_memory_gb * 0.5 * 1e9) as usize;

    // Calculate max batch size
    let max_batch = usable_memory / bytes_per_sample;

    // Round down to power of 2 for efficiency
    let batch_size = (max_batch as f64).log2().floor() as u32;
    let batch_size = 2usize.pow(batch_size).min(512).max(16);

    Ok(batch_size)
}

// ============================================================================
// PYTHON MODULE DEFINITION
// ============================================================================

#[pymodule]
fn tmt_rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    // Fractional Differentiation
    m.add_function(wrap_pyfunction!(frac_diff_ffd, m)?)?;
    m.add_function(wrap_pyfunction!(frac_diff_ffd_batch, m)?)?;

    // Rolling Statistics
    m.add_function(wrap_pyfunction!(rolling_mean, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_var, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_stats_batch, m)?)?;

    // Sequence Creation
    m.add_function(wrap_pyfunction!(create_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(create_sequence_indices, m)?)?;

    // Technical Indicators
    m.add_function(wrap_pyfunction!(compute_rsi, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ema, m)?)?;
    m.add_function(wrap_pyfunction!(compute_macd, m)?)?;
    m.add_function(wrap_pyfunction!(compute_bollinger_bands, m)?)?;

    // Information Theory
    m.add_function(wrap_pyfunction!(rolling_entropy, m)?)?;

    // Market Microstructure
    m.add_function(wrap_pyfunction!(compute_order_flow_imbalance, m)?)?;
    m.add_function(wrap_pyfunction!(compute_quote_pressure, m)?)?;

    // Performance Metrics
    m.add_function(wrap_pyfunction!(compute_ic, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sharpe_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_drawdown, m)?)?;

    // Memory Utilities
    m.add_function(wrap_pyfunction!(suggest_batch_size, m)?)?;

    Ok(())
}
