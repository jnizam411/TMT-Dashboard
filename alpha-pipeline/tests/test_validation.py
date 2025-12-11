"""Tests for validation module."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.validation.purged_walk_forward import PurgedWalkForwardSplitter, PurgedKFold
from src.validation.metrics import (
    compute_ic,
    compute_daily_ic,
    compute_icir,
    compute_sharpe,
    compute_max_drawdown,
    compute_turnover,
)


@pytest.fixture
def sample_predictions():
    """Generate sample predictions and returns."""
    np.random.seed(42)
    n = 252
    dates = pd.bdate_range("2020-01-01", periods=n)

    predictions = pd.DataFrame(
        np.random.randn(n, 50),
        index=dates,
        columns=[f"stock_{i}" for i in range(50)],
    )

    # Forward returns with some correlation to predictions
    noise = np.random.randn(n, 50) * 0.02
    returns = predictions * 0.001 + noise  # Small positive correlation

    return predictions, returns


class TestPurgedWalkForward:
    """Tests for purged walk-forward splitter."""

    def test_generate_splits(self):
        """Test split generation."""
        splitter = PurgedWalkForwardSplitter(
            train_start=date(2005, 1, 1),
            train_end=date(2015, 12, 31),
            purge_days=63,
            val_days=252,
            test_days=252,
            roll_days=252,
            final_test_end=date(2025, 12, 31),
        )

        splits = splitter.generate_splits()

        assert len(splits) > 0
        assert splits[0].split_id == 0

    def test_no_overlap(self):
        """Test that train and test don't overlap."""
        splitter = PurgedWalkForwardSplitter(
            train_start=date(2010, 1, 1),
            train_end=date(2015, 12, 31),
            purge_days=63,
            val_days=126,
            test_days=126,
        )

        splits = splitter.generate_splits()

        for split in splits:
            # Train end + purge < val start
            assert split.train_end < split.val_start
            assert split.val_end < split.test_start

    def test_purge_gap(self):
        """Test that purge gap is respected."""
        splitter = PurgedWalkForwardSplitter(
            purge_days=63,
        )

        splits = splitter.generate_splits()

        for split in splits:
            # Check purge gap
            trading_days = pd.bdate_range(split.train_end, split.val_start)
            assert len(trading_days) >= 63

    def test_summary(self):
        """Test summary DataFrame."""
        splitter = PurgedWalkForwardSplitter()
        splitter.generate_splits()

        summary = splitter.summary()

        assert "split_id" in summary.columns
        assert "train_start" in summary.columns
        assert "test_end" in summary.columns


class TestPurgedKFold:
    """Tests for purged K-Fold."""

    def test_n_splits(self, sample_predictions):
        """Test number of splits."""
        predictions, returns = sample_predictions

        kfold = PurgedKFold(n_splits=5, purge_days=21)

        splits = list(kfold.split(predictions))

        assert len(splits) == 5

    def test_no_leakage(self, sample_predictions):
        """Test that there's no data leakage."""
        predictions, returns = sample_predictions

        kfold = PurgedKFold(n_splits=5, purge_days=21)

        for train_idx, test_idx in kfold.split(predictions):
            # No overlap between train and test indices
            assert len(set(train_idx) & set(test_idx)) == 0


class TestMetrics:
    """Tests for performance metrics."""

    def test_compute_ic(self, sample_predictions):
        """Test IC computation."""
        predictions, returns = sample_predictions

        # Flatten for single IC
        pred_flat = predictions.values.flatten()
        ret_flat = returns.values.flatten()

        ic = compute_ic(
            pd.Series(pred_flat),
            pd.Series(ret_flat),
        )

        assert -1 <= ic <= 1

    def test_compute_daily_ic(self, sample_predictions):
        """Test daily IC computation."""
        predictions, returns = sample_predictions

        daily_ic = compute_daily_ic(predictions, returns)

        assert len(daily_ic) == len(predictions)
        assert (daily_ic.abs() <= 1).all()

    def test_compute_icir(self, sample_predictions):
        """Test ICIR computation."""
        predictions, returns = sample_predictions

        daily_ic = compute_daily_ic(predictions, returns)
        icir = compute_icir(daily_ic)

        assert np.isfinite(icir)

    def test_compute_sharpe(self):
        """Test Sharpe ratio computation."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)

        sharpe = compute_sharpe(returns)

        assert np.isfinite(sharpe)

    def test_compute_max_drawdown(self):
        """Test max drawdown computation."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.02)

        max_dd = compute_max_drawdown(returns)

        assert max_dd <= 0  # Drawdown is negative
        assert max_dd >= -1  # Can't lose more than 100%

    def test_compute_turnover(self, sample_predictions):
        """Test turnover computation."""
        predictions, returns = sample_predictions

        # Create weight matrix
        weights = predictions.apply(
            lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"),
            axis=1,
        )
        weights = (weights == 4).astype(float) / (weights == 4).sum(axis=1).values[:, None]
        weights = weights.fillna(0)

        turnover = compute_turnover(weights)

        assert (turnover >= 0).all()
        assert (turnover <= 2).all()  # Max 200% turnover
