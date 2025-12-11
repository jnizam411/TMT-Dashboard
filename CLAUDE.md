# CLAUDE.md - AI Assistant Guide for TMT-Dashboard

## System Prompt

You are a world-class senior software engineer who ships reliable, maintainable production code at companies like OpenAI, Tesla, and Stripe.

### Core Philosophy (Never Violate These)

1. **With traditional software**, anything you can fully specify, you can automate perfectly.
2. **With AI**, anything you can rigorously verify, you can automate reliably.
→ **Therefore**, your job is to turn even vague user requests into fully specified, automatically verifiable deliverables.

### Workflow (Always Follow in This Exact Order)

#### 1. Think Step-by-Step and Clarify
- Restate the user's goal in precise, unambiguous language.
- Explicitly list every assumption you are making.
- Ask for clarification on anything ambiguous before writing a single line of code.

#### 2. Design with Verifiability Baked In from the Start
- Choose the simplest possible architecture that solves the problem.
- Explicitly define success criteria (performance, edge cases, input/output examples).
- Write a concrete verification plan (unit tests, property-based tests, type signatures, example runs, benchmarks, linting, security checks).

#### 3. Write the Code
- Use current stable versions of languages and libraries (2025 best practices).
- Prefer explicit over implicit, boring over clever.
- Add comprehensive docstrings and inline comments only where non-obvious.
- Never hallucinate APIs — if unsure, say so and ask.

#### 4. Make It Self-Verifying
- Every file you create must include a complete test suite that can be run with one command and achieves ≥95% coverage on meaningful paths.
- For scripts/notebooks: include a "smoke test" section at the bottom that runs on import.
- For libraries/apps: include pytest/playwright/cypress tests + CI GitHub Actions workflow if applicable.
- For data/science tasks: include statistical checks, assertion on output shapes, and reproducibility seed.

#### 5. Deliver in Ready-to-Run Format
- Use proper file structure and separate files with clear names.
- Include a short README section at the top of the main response explaining exactly how to run and verify everything.
- End every response with a verification summary showing test results, coverage, and validation checks passed.

---

## Repository Overview

**TMT-Dashboard** is an institutional-grade quantitative finance system for equity alpha generation and prediction. The repository contains two main components:

1. **alpha-pipeline**: Enterprise-grade equity alpha generation pipeline with rigorous backtesting
2. **tmt_optimized**: Apple Silicon-optimized financial ML library with Rust acceleration

### Key Characteristics
- **Domain**: Quantitative Finance / Machine Learning
- **Language Stack**: Python 3.11+, Rust (for acceleration)
- **ML Framework**: PyTorch 2.4+ with Metal Performance Shaders (MPS) support
- **Hardware Target**: Apple M-series (M1/M2/M3/M4) with fallback to CPU
- **Production Grade**: Includes comprehensive testing, validation, and backtesting

---

## Repository Structure

```
TMT-Dashboard/
├── CLAUDE.md                      # This file - AI assistant guide
├── CODEBASE_EVALUATION.md         # Detailed technical evaluation
├── Pytorch_test.ipynb             # Original LSTM implementation (600KB)
├── Pytorch_optimized.ipynb        # Optimized notebook
│
├── alpha-pipeline/                # Enterprise alpha generation system
│   ├── run_pipeline.py            # Main entry point (8 phases)
│   ├── run_live.py                # Live inference entry point
│   ├── requirements.txt           # Pinned dependencies
│   ├── pyproject.toml             # Package configuration
│   ├── environment.yml            # Conda environment
│   │
│   ├── src/                       # Source code
│   │   ├── config/                # Configuration management
│   │   │   ├── settings.py        # Pydantic settings
│   │   │   └── __init__.py
│   │   │
│   │   ├── data/                  # Data acquisition
│   │   │   ├── downloader.py      # yfinance integration
│   │   │   ├── universe.py        # Universe construction
│   │   │   └── __init__.py
│   │   │
│   │   ├── features/              # Feature engineering
│   │   │   ├── pipeline.py        # Main feature pipeline
│   │   │   ├── fracdiff.py        # Fractional differentiation
│   │   │   ├── microstructure.py  # Order flow features
│   │   │   ├── csi_vpin.py        # VPIN calculation
│   │   │   ├── technical.py       # Technical indicators
│   │   │   └── __init__.py
│   │   │
│   │   ├── labeling/              # Label generation
│   │   │   ├── triple_barrier.py  # Triple-barrier method
│   │   │   ├── meta_labeling.py   # Meta-labeling
│   │   │   └── __init__.py
│   │   │
│   │   ├── validation/            # Cross-validation
│   │   │   ├── purged_walk_forward.py  # CPCV implementation
│   │   │   ├── metrics.py         # Financial metrics
│   │   │   └── __init__.py
│   │   │
│   │   ├── models/                # ML models
│   │   │   ├── lgbm.py            # LightGBM wrapper
│   │   │   ├── lasso.py           # Feature selection
│   │   │   ├── ensemble.py        # Model ensembling
│   │   │   └── __init__.py
│   │   │
│   │   ├── backtest/              # Backtesting
│   │   │   ├── vectorized.py      # Vectorized backtest
│   │   │   └── __init__.py
│   │   │
│   │   └── inference/             # Live inference
│   │       ├── pipeline.py        # Inference pipeline
│   │       └── __init__.py
│   │
│   ├── notebooks/                 # Analysis notebooks
│   │   ├── 00_sanity_checks.ipynb
│   │   ├── 01_feature_exploration.ipynb
│   │   └── 99_final_backtest.ipynb
│   │
│   └── tests/                     # Test suite
│       ├── test_features.py
│       ├── test_labeling.py
│       ├── test_validation.py
│       └── test_backtest.py
│
└── tmt_optimized/                 # Apple Silicon optimization
    ├── README.md                  # Hardware optimization guide
    ├── requirements.txt           # Core dependencies
    ├── pyproject.toml             # Package config
    ├── __init__.py                # Package exports
    │
    ├── mps_accelerator.py         # MPS/Metal GPU integration
    ├── memory_manager.py          # Adaptive memory management
    ├── feature_engine.py          # Optimized feature engineering
    ├── lstm_model.py              # MPS-accelerated LSTM
    │
    └── rust_core/                 # Rust acceleration
        ├── Cargo.toml             # Rust dependencies
        ├── build.sh               # Build script
        └── src/
            └── lib.rs             # SIMD implementations
```

---

## Development Workflows

### Alpha Pipeline Workflow

The alpha-pipeline follows an 8-phase production workflow:

```bash
# Phase 1-8: Full pipeline
python alpha-pipeline/run_pipeline.py --mode full

# Individual phases
python alpha-pipeline/run_pipeline.py --mode data       # Data acquisition
python alpha-pipeline/run_pipeline.py --mode features   # Feature engineering
python alpha-pipeline/run_pipeline.py --mode labels     # Label generation
python alpha-pipeline/run_pipeline.py --mode train      # Model training
python alpha-pipeline/run_pipeline.py --mode backtest   # Backtesting

# Live inference
python alpha-pipeline/run_live.py
```

#### Phase Sequence (Always Follow)

1. **Data Acquisition**: Download universe + price data + macro series
2. **Labeling**: Triple-barrier method + meta-labeling
3. **Feature Engineering**: 45+ features (FFD, microstructure, technical)
4. **Cross-Sectional Processing**: Market neutralization + z-score normalization
5. **Validation Setup**: Purged walk-forward splits (CPCV)
6. **Model Training**: 2-stage cascaded models (barrier touch → direction)
7. **Ensemble**: Combine Stage 1 + Stage 2 predictions
8. **Backtesting**: Vectorized backtest with transaction costs

### TMT Optimized Workflow

```python
# Basic usage
from tmt_optimized import (
    MPSAccelerator,
    OptimizedFeatureEngine,
    OptimizedLSTM,
    MPSTrainer,
)

# 1. Initialize hardware accelerator
accelerator = MPSAccelerator()  # Auto-detects MPS/CPU

# 2. Create feature engine (uses Rust if available)
engine = OptimizedFeatureEngine()
features = engine.engineer_all_features(df)

# 3. Create sequences
X_seq, y_seq = engine.create_sequences(X, y, seq_length=30)

# 4. Train with memory-aware settings
trainer = MPSTrainer(model, training_config, accelerator, memory_manager)
history = trainer.train(X_train, y_train, X_val, y_val)
```

### Testing Workflow

```bash
# Run all tests
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test module
pytest tests/test_features.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

---

## Key Conventions

### Code Style

**Python**
- **Formatter**: Black (line length: 100)
- **Import Sorting**: isort (black profile)
- **Linter**: Ruff (E, F, W, I, N, B, C4)
- **Type Checker**: mypy with strict mode
- **Python Version**: 3.11+ (3.9+ for tmt_optimized)

**Configuration Files**
- All configs in `pyproject.toml` (no setup.py)
- Use `structlog` for structured logging
- Pydantic for configuration management

### Naming Conventions

```python
# Classes: PascalCase
class OptimizedLSTM:
    pass

class MPSAccelerator:
    pass

# Functions/methods: snake_case
def compute_sharpe_ratio(returns: np.ndarray) -> float:
    pass

def frac_diff_ffd(series: pd.Series, d: float) -> pd.Series:
    pass

# Constants: UPPER_SNAKE_CASE
TRADING_DAYS_PER_YEAR = 252
DEFAULT_BATCH_SIZE = 64

# Private methods: _leading_underscore
def _validate_input(self, data: pd.DataFrame) -> None:
    pass
```

### Import Order

```python
# 1. Standard library
import sys
from datetime import date
from pathlib import Path
from typing import Optional

# 2. Third-party packages
import numpy as np
import pandas as pd
import torch
import structlog

# 3. Local imports
from src.config import settings
from src.features import FeatureFactory
```

### Documentation Style

```python
def compute_ic(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Compute Information Coefficient (Spearman rank correlation).

    The IC measures the quality of an alpha signal by computing the
    rank correlation between predictions and actual returns.

    Args:
        predictions: Model predictions, shape (n_samples,)
        actuals: Actual returns, shape (n_samples,)

    Returns:
        Information Coefficient in range [-1, 1]

    Raises:
        ValueError: If arrays have different lengths or contain NaN

    Example:
        >>> preds = np.array([0.1, 0.2, -0.1])
        >>> actual = np.array([0.05, 0.15, -0.05])
        >>> ic = compute_ic(preds, actual)
        >>> assert -1 <= ic <= 1
    """
    pass
```

---

## Financial ML Best Practices

### Feature Engineering Principles

1. **Always achieve stationarity first**
   - Use Fractional Differentiation (FFD) for d ∈ [0.3, 0.5]
   - Never use raw prices directly
   - Verify stationarity with ADF test

2. **Market neutralization**
   - Remove market beta using rolling OLS
   - Window size: 60 days minimum
   - Use robust regression when outliers present

3. **Cross-sectional normalization**
   - Z-score normalization within each date
   - Winsorize at ±3σ to handle outliers
   - Apply after all transformations

4. **Feature types (45+ total)**
   - **Microstructure**: Order flow imbalance, quote pressure, VWAP distance
   - **Technical**: RSI, MACD, Bollinger Bands, ATR
   - **Volatility**: Parkinson, Garman-Klass, realized volatility
   - **Information Theory**: Shannon entropy, sample entropy
   - **Regime**: HMM-based regime detection (3 states)
   - **Intermarket**: Correlation with SPY, QQQ, VIX, TLT, GLD

### Labeling Best Practices

**Triple-Barrier Method**
```python
# Standard parameters
profit_take = 2.0  # 2% profit target
stop_loss = 1.0    # 1% stop loss
max_hold = 5       # 5-day maximum holding period

# Meta-labeling (2-stage approach)
# Stage 1: Predict if barrier will be touched (classification)
# Stage 2: Predict direction given touch (classification on filtered samples)
```

### Validation Principles

**Purged Combinatorial Cross-Validation (CPCV)**
- **Purging**: Remove samples within ±5 days of test set
- **Embargo**: Additional 2-day embargo after test set
- **Splits**: 10 walk-forward splits minimum
- **Group by**: Date (prevent data leakage)

**Financial Metrics (Use These)**
```python
# Signal quality
ic = compute_ic(predictions, actuals)  # Target: |IC| > 0.05

# Risk-adjusted returns
sharpe_ratio = compute_sharpe_ratio(returns)  # Target: Sharpe > 1.0
max_drawdown = compute_max_drawdown(returns)  # Target: MDD < 20%

# Accuracy metrics (less important)
accuracy = (predictions > 0) == (actuals > 0)  # Target: > 52%
```

---

## Performance Optimization Guidelines

### Hardware-Specific Optimizations

**Apple Silicon (M1/M2/M3/M4)**
```python
# 1. Always check MPS availability
import torch
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# 2. Use unified memory efficiently
# - No explicit GPU memory management needed
# - Avoid frequent CPU↔GPU transfers
# - Batch operations when possible

# 3. Leverage Rust acceleration for CPU-bound tasks
# - Fractional differentiation: 20x speedup
# - Rolling statistics: 13x speedup
# - Technical indicators: 15x speedup
```

**Memory Management**
```python
# Use adaptive memory manager
from tmt_optimized import AdaptiveMemoryManager, optimize_for_memory

# Get recommended settings
settings = optimize_for_memory(available_gb=16)
# Returns: {'batch_size': 64, 'hidden_size': 128, ...}

# Dynamic batch sizing
batch_size = memory_manager.get_dynamic_batch_size(
    sample_shape=(30, 45),
    model_memory_gb=0.5,
)
```

### Performance Targets

| Operation | Target | Method |
|-----------|--------|--------|
| Feature Engineering (3741 samples) | < 5s | Rust + SIMD |
| LSTM Training (50 epochs) | < 60s | MPS acceleration |
| Backtest (5 years daily) | < 10s | Vectorized operations |
| Full Pipeline (data → predictions) | < 5min | Parallel processing |

### Optimization Priority

1. **Critical Path** (optimize first):
   - Fractional differentiation (convolution)
   - Rolling window operations
   - Sequence creation

2. **Medium Priority**:
   - Technical indicator computation
   - HMM training
   - Hyperparameter optimization parallelization

3. **Low Priority** (already optimized):
   - LSTM forward/backward (cuDNN/MPS)
   - Matrix operations (BLAS)

---

## Git Workflow

### Branch Naming Convention

```
claude/<description>-<session-id>

Examples:
- claude/equity-alpha-pipeline-01SFAMbBSv6PME4om6DHtWqp
- claude/evaluate-codebase-optimization-019FaZX3MQ8Mq6tc7mWMUSuX
- claude/claude-md-mj1qz4re40ktktuq-01PGvG16dHiPAjAkVbLBFFn4
```

### Commit Message Style

Analyze recent commits to match style:

```bash
git log --oneline -10

# Observed patterns:
# - "Add enterprise-grade equity alpha generation pipeline"
# - "Add Apple M-series hardware optimization with Rust acceleration"
# - "Add comprehensive codebase evaluation"
# - "Merge pull request #N from user/branch-name"
```

**Format**:
- Imperative mood: "Add", "Fix", "Update", "Refactor"
- Descriptive: Clearly state what was added/changed
- Concise: Single line for simple changes
- Multi-line: Use body for complex changes

### Git Operations Protocol

**Push with Retry**
```bash
# Always use -u flag for new branches
git push -u origin <branch-name>

# If network failure, retry up to 4 times with exponential backoff
# Wait times: 2s, 4s, 8s, 16s
```

**Pull/Fetch**
```bash
# Prefer specific branch fetching
git fetch origin <branch-name>
git pull origin <branch-name>

# Apply same retry logic as push
```

**Safety Rules**
- NEVER push to main/master without PR
- NEVER force push unless explicitly requested
- NEVER skip hooks (--no-verify)
- ALWAYS create PRs for merging to main

---

## Testing Standards

### Test Organization

```
tests/
├── test_features.py         # Feature engineering tests
├── test_labeling.py         # Labeling logic tests
├── test_validation.py       # Cross-validation tests
├── test_backtest.py         # Backtesting tests
└── __init__.py
```

### Test Patterns

```python
import pytest
import numpy as np
import pandas as pd
from src.features.fracdiff import frac_diff_ffd

class TestFractionalDifferentiation:
    """Test fractional differentiation implementation."""

    def test_ffd_basic(self):
        """Test FFD with standard parameters."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = frac_diff_ffd(series, d=0.4)

        assert isinstance(result, pd.Series)
        assert len(result) <= len(series)
        assert not result.isna().all()

    def test_ffd_stationarity(self):
        """Verify FFD achieves stationarity."""
        # Generate non-stationary series
        series = pd.Series(np.cumsum(np.random.randn(1000)))

        # Apply FFD
        result = frac_diff_ffd(series, d=0.5)

        # Check stationarity with ADF test
        from statsmodels.tsa.stattools import adfuller
        adf_stat, p_value, *_ = adfuller(result.dropna())

        assert p_value < 0.05  # Reject null (non-stationary)

    @pytest.mark.parametrize("d", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_ffd_different_d_values(self, d):
        """Test FFD with various d values."""
        series = pd.Series(np.random.randn(100))
        result = frac_diff_ffd(series, d=d)

        assert isinstance(result, pd.Series)
        assert len(result) > 0
```

### Coverage Requirements

- **Minimum Coverage**: 80% overall
- **Critical Paths**: 95%+ (feature engineering, labeling, validation)
- **Exception Handling**: All error paths tested
- **Edge Cases**: Empty inputs, NaN handling, extreme values

### Running Tests

```bash
# All tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Specific module
pytest tests/test_features.py::TestFractionalDifferentiation -v

# With markers
pytest -m "not slow" -v  # Skip slow tests

# Generate HTML coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Common Issues and Solutions

### MPS/Metal Issues

**Problem**: MPS not available
```python
# Check MPS status
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Requirements:
# - PyTorch 2.0+
# - macOS 12.3+
# - Apple Silicon (M1/M2/M3/M4)
```

**Problem**: MPS numerical instability
```python
# Solution: Use float32 instead of float16
model = model.to(dtype=torch.float32)

# Or fall back to CPU for problematic operations
if not stable_on_mps:
    tensor = tensor.cpu()
```

### Memory Issues

**Problem**: Out of memory during training
```python
# Solution 1: Reduce batch size
trainer.config.batch_size = 16

# Solution 2: Enable gradient accumulation
trainer.config.gradient_accumulation_steps = 4

# Solution 3: Use memory manager
memory_manager.clear_memory()
torch.mps.empty_cache()  # Clear MPS cache

# Solution 4: Enable checkpointing
memory_manager.enable_checkpointing = True
```

### Rust Build Issues

**Problem**: Rust compilation fails
```bash
# Check Rust installation
rustc --version

# Update Rust
rustup update

# Rebuild with verbose output
cd tmt_optimized/rust_core
RUST_BACKTRACE=1 maturin develop --release

# If maturin not found
pip install maturin

# Force rebuild
cargo clean
./build.sh
```

### Data Issues

**Problem**: Yahoo Finance rate limiting
```python
# Solution: Add delays and caching
import yfinance as yf
import time

def download_with_retry(ticker, max_retries=3):
    for i in range(max_retries):
        try:
            data = yf.download(ticker, progress=False)
            return data
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(2 ** i)  # Exponential backoff
            else:
                raise
```

### Validation Issues

**Problem**: Data leakage in cross-validation
```python
# WRONG: Standard sklearn cross-validation
from sklearn.model_selection import KFold
cv = KFold(n_splits=5)  # ❌ Leakage!

# CORRECT: Purged walk-forward with embargo
from src.validation import PurgedWalkForwardSplitter
splitter = PurgedWalkForwardSplitter(
    purge_days=5,
    embargo_days=2,
)
splits = splitter.generate_splits()
```

---

## Security Best Practices

### API Keys and Secrets

```python
# NEVER hardcode secrets
API_KEY = "sk-1234567890"  # ❌ WRONG

# Use environment variables
import os
API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")  # ✅ CORRECT

# Or use .env file (add to .gitignore)
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
```

### Input Validation

```python
# Always validate external inputs
def download_ticker(ticker: str) -> pd.DataFrame:
    # Validate ticker format
    if not ticker.isalnum() or len(ticker) > 5:
        raise ValueError(f"Invalid ticker format: {ticker}")

    # Sanitize before API call
    ticker = ticker.upper().strip()

    return yf.download(ticker)
```

### Dependency Security

```bash
# Pin all dependencies (already done)
# See requirements.txt and pyproject.toml

# Audit dependencies periodically
pip install pip-audit
pip-audit

# Update dependencies carefully
pip list --outdated
# Test thoroughly after updates
```

---

## Performance Benchmarks

### Expected Performance (Apple M2 Pro, 16GB RAM)

| Pipeline Stage | Target Time | Bottleneck |
|----------------|-------------|------------|
| Data Download (50 tickers) | 30-60s | Network I/O |
| Feature Engineering | < 5s | CPU (Rust accelerated) |
| Sequence Creation | < 0.2s | CPU (Rust accelerated) |
| LSTM Training (50 epochs) | 40-60s | GPU (MPS) |
| Hyperparameter Optimization (100 trials) | 60-120min | GPU + CPU |
| Backtesting (5 years) | < 10s | CPU (vectorized) |

### Rust Acceleration Speedups

| Operation | Python (NumPy) | Rust + SIMD | Speedup |
|-----------|----------------|-------------|---------|
| FFD Convolution | ~500ms | ~25ms | **20x** |
| Rolling Mean | ~100ms | ~8ms | **12x** |
| Rolling Std | ~150ms | ~12ms | **13x** |
| RSI | ~80ms | ~5ms | **16x** |
| MACD | ~120ms | ~8ms | **15x** |
| Bollinger Bands | ~90ms | ~6ms | **15x** |
| Sharpe Ratio | ~20ms | ~2ms | **10x** |
| Max Drawdown | ~30ms | ~3ms | **10x** |

### Full Pipeline Benchmark

```
Original (Python only):    ~182s
Optimized (Rust + MPS):     ~58s
Speedup:                    3.1x

Breakdown:
- Data Download:     15s → 15s (1.0x) - Network bound
- Features:          45s → 3s  (15x) - Rust acceleration
- Sequences:         2s  → 0.1s (20x) - Rust acceleration
- Training:          120s → 40s (3x)  - MPS acceleration
```

---

## Quick Reference

### Essential Commands

```bash
# Alpha Pipeline
python alpha-pipeline/run_pipeline.py --mode full
python alpha-pipeline/run_pipeline.py --mode backtest
python alpha-pipeline/run_live.py

# Testing
pytest tests/ -v --cov=src
pytest tests/test_features.py -v

# Code Quality
black . --check
isort . --check-only
ruff check .
mypy src/

# Rust Build
cd tmt_optimized/rust_core
./build.sh
maturin develop --release

# Git
git fetch origin <branch>
git push -u origin <branch>
git log --oneline -10
```

### Key Files to Check

```
alpha-pipeline/run_pipeline.py    # Main entry point
alpha-pipeline/src/features/      # Feature engineering
alpha-pipeline/src/validation/    # CPCV implementation
tmt_optimized/mps_accelerator.py  # MPS integration
tmt_optimized/rust_core/src/      # Rust implementations
CODEBASE_EVALUATION.md            # Detailed analysis
```

### Important Constants

```python
TRADING_DAYS_PER_YEAR = 252
FFD_D_VALUES = [0.3, 0.4, 0.5]  # Stationarity vs memory tradeoff
DEFAULT_BATCH_SIZE = 64
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_SEQ_LENGTH = 30
PURGE_DAYS = 5
EMBARGO_DAYS = 2
MIN_IC_THRESHOLD = 0.05
TARGET_SHARPE = 1.0
```

---

## Resources

### Internal Documentation
- `CODEBASE_EVALUATION.md` - Technical deep dive
- `tmt_optimized/README.md` - Hardware optimization guide
- `alpha-pipeline/notebooks/` - Example workflows

### External References
- **Advances in Financial Machine Learning** (Marcos López de Prado)
  - Fractional Differentiation (Chapter 5)
  - Labeling (Chapter 3)
  - Backtesting (Chapter 7)

- **Machine Learning for Algorithmic Trading** (Stefan Jansen)
  - Feature engineering
  - Cross-validation

- **PyTorch MPS Documentation**: https://pytorch.org/docs/stable/notes/mps.html
- **Rust + Python Integration**: https://pyo3.rs/

---

## Changelog

### 2025-12-11
- Initial CLAUDE.md creation
- Added comprehensive system prompt
- Documented repository structure
- Added performance benchmarks
- Included testing standards

---

## Contact and Support

For questions about this codebase:
1. Check `CODEBASE_EVALUATION.md` for technical details
2. Review `tmt_optimized/README.md` for optimization guide
3. Examine test files in `tests/` for usage examples
4. Review notebooks in `alpha-pipeline/notebooks/` for workflows

---

**Last Updated**: December 11, 2025
**Repository**: TMT-Dashboard
**Version**: 1.0.0
