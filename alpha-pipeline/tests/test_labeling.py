"""Tests for labeling module."""

import numpy as np
import pandas as pd
import pytest

from src.labeling.triple_barrier import (
    TripleBarrierLabeler,
    get_barrier_events,
    compute_daily_volatility,
)
from src.labeling.meta_labeling import MetaLabeler, get_meta_labels


@pytest.fixture
def sample_prices():
    """Generate sample price data."""
    np.random.seed(42)
    n = 252  # 1 year
    dates = pd.bdate_range("2020-01-01", periods=n)

    # Random walk with drift
    returns = np.random.randn(n) * 0.02 + 0.0005
    close = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "close": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "open": close * 1.001,
        "volume": np.random.exponential(1_000_000, n),
    }, index=dates)

    return df


class TestTripleBarrier:
    """Tests for triple barrier labeling."""

    def test_compute_volatility(self, sample_prices):
        """Test volatility computation."""
        vol = compute_daily_volatility(sample_prices["close"], lookback=20)

        assert len(vol) == len(sample_prices)
        assert vol.iloc[:20].isna().all()  # First 20 should be NaN
        assert vol.iloc[20:].notna().all()
        assert (vol > 0).all()  # All positive

    def test_barrier_events_basic(self, sample_prices):
        """Test basic barrier event generation."""
        labeler = TripleBarrierLabeler(
            horizontal_barrier_days=10,
            vertical_barrier_multiplier=1.5,
            vol_lookback=20,
        )

        events = labeler.get_barrier_events(sample_prices)

        assert len(events) > 0
        assert "barrier_side" in events.columns
        assert "return_at_touch" in events.columns
        assert "label" in events.columns

    def test_barrier_sides(self, sample_prices):
        """Test that barrier sides are correct."""
        events = get_barrier_events(sample_prices)

        # Should have all three types
        barrier_types = events["barrier_side"].unique()
        assert len(barrier_types) > 1  # At least 2 types

        # Values should be -1, 0, or 1
        assert all(b in [-1, 0, 1] for b in barrier_types)

    def test_labels_match_barriers(self, sample_prices):
        """Test that labels match barrier touches."""
        events = get_barrier_events(sample_prices)

        # Upper barrier -> label should be +1
        upper_mask = events["barrier_side"] == 1
        assert (events.loc[upper_mask, "label"] == 1).all()

        # Lower barrier -> label should be -1
        lower_mask = events["barrier_side"] == -1
        assert (events.loc[lower_mask, "label"] == -1).all()

    def test_t0_before_t1(self, sample_prices):
        """Test that t0 is always before t1."""
        events = get_barrier_events(sample_prices)

        events = events.reset_index()
        assert (events["t0"] <= events["t1"]).all()

    def test_days_held_positive(self, sample_prices):
        """Test that days held is positive."""
        events = get_barrier_events(sample_prices)

        assert (events["days_held"] >= 0).all()
        assert (events["days_held"] <= 10).all()  # Max is horizontal barrier


class TestMetaLabeling:
    """Tests for meta-labeling."""

    def test_stage1_labels(self, sample_prices):
        """Test Stage 1 label creation."""
        events = get_barrier_events(sample_prices)
        meta = MetaLabeler()

        stage1 = meta.create_stage1_labels(events)

        assert len(stage1) == len(events)
        assert stage1.isin([0, 1]).all()

        # Should be 1 when barrier touched (not time barrier)
        touched = events["barrier_side"] != 0
        assert (stage1[touched] == 1).all()

    def test_stage2_labels(self, sample_prices):
        """Test Stage 2 label creation."""
        events = get_barrier_events(sample_prices)
        meta = MetaLabeler(stage1_threshold=0.5)  # Lower threshold for test

        # Simulate Stage 1 predictions
        stage1_probs = pd.Series(
            np.random.uniform(0.3, 0.9, len(events)),
            index=events.index,
        )

        stage2 = meta.create_stage2_labels(events, stage1_probs)

        # Should only have high-confidence samples
        assert len(stage2) <= len(events)
        assert (stage2["stage1_prob"] > meta.stage1_threshold).all()

    def test_get_meta_labels(self, sample_prices):
        """Test meta-label generation."""
        events = get_barrier_events(sample_prices)

        # Simulate Stage 1 predictions
        stage1_probs = pd.Series(
            np.random.uniform(0.3, 0.9, len(events)),
            index=events.index,
        )

        meta_labels = get_meta_labels(events, stage1_probs, stage1_threshold=0.6)

        # Should have NaN for low-confidence
        assert meta_labels.isna().any()

        # Non-NaN should be +1 or -1
        valid = meta_labels.dropna()
        assert valid.isin([1, -1]).all()


class TestSampleWeights:
    """Tests for sample weight computation."""

    def test_weights_sum_to_one(self, sample_prices):
        """Test that sample weights sum to 1."""
        labeler = TripleBarrierLabeler()
        events = labeler.get_barrier_events(sample_prices)

        weights = labeler.get_sample_weights(events, sample_prices["close"])

        assert abs(weights.sum() - 1.0) < 1e-6

    def test_weights_positive(self, sample_prices):
        """Test that all weights are positive."""
        labeler = TripleBarrierLabeler()
        events = labeler.get_barrier_events(sample_prices)

        weights = labeler.get_sample_weights(events, sample_prices["close"])

        assert (weights > 0).all()
