"""Tests for backtesting module."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import TransactionCostModel, MarketImpactModel
from src.backtest.vectorized import VectorizedBacktest


@pytest.fixture
def sample_signals():
    """Generate sample signals and returns."""
    np.random.seed(42)
    n_days = 252
    n_stocks = 100

    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"stock_{i}" for i in range(n_stocks)]

    # Random signals
    signals = pd.DataFrame(
        np.random.randn(n_days, n_stocks),
        index=dates,
        columns=tickers,
    )

    # Returns with small positive correlation to signals
    noise = np.random.randn(n_days, n_stocks) * 0.02
    returns = signals * 0.0005 + noise

    # ADV
    adv = pd.DataFrame(
        np.random.exponential(10_000_000, (n_days, n_stocks)),
        index=dates,
        columns=tickers,
    )

    return signals, returns, adv


class TestTransactionCosts:
    """Tests for transaction cost model."""

    def test_fixed_cost(self):
        """Test fixed cost computation."""
        model = TransactionCostModel(fixed_cost_bps=10.0)
        cost = model.compute_cost(1_000_000)

        # 10 bps on $1M = $1000
        assert cost.fixed_cost == 1000

    def test_spread_cost(self):
        """Test spread cost computation."""
        model = TransactionCostModel(
            fixed_cost_bps=0,
            spread_cost_bps=5.0,
        )
        cost = model.compute_cost(1_000_000)

        # 5 bps on $1M = $500
        assert cost.spread_cost == 500

    def test_total_cost(self):
        """Test total cost is sum of components."""
        model = TransactionCostModel(
            fixed_cost_bps=10.0,
            spread_cost_bps=5.0,
        )
        cost = model.compute_cost(1_000_000)

        assert cost.total_cost == cost.fixed_cost + cost.spread_cost + cost.impact_cost


class TestMarketImpact:
    """Tests for market impact model."""

    def test_basic_impact(self):
        """Test basic impact computation."""
        model = MarketImpactModel(impact_bps_per_mm=0.1)

        impact = model.compute_impact(
            trade_value=1_000_000,
            adv=10_000_000,
        )

        assert impact > 0
        assert np.isfinite(impact)

    def test_impact_scales_with_size(self):
        """Test that impact scales with trade size."""
        model = MarketImpactModel()

        small_impact = model.compute_impact(100_000, 10_000_000)
        large_impact = model.compute_impact(1_000_000, 10_000_000)

        assert large_impact > small_impact

    def test_impact_scales_with_adv(self):
        """Test that impact scales inversely with ADV."""
        model = MarketImpactModel()

        liquid_impact = model.compute_impact(1_000_000, 100_000_000)
        illiquid_impact = model.compute_impact(1_000_000, 10_000_000)

        assert illiquid_impact > liquid_impact


class TestVectorizedBacktest:
    """Tests for vectorized backtester."""

    def test_basic_backtest(self, sample_signals):
        """Test basic backtest runs."""
        signals, returns, adv = sample_signals

        backtester = VectorizedBacktest()
        result = backtester.run(signals, returns, adv)

        assert result.gross_sharpe is not None
        assert result.net_sharpe is not None
        assert len(result.net_returns) > 0

    def test_costs_reduce_sharpe(self, sample_signals):
        """Test that costs reduce Sharpe ratio."""
        signals, returns, adv = sample_signals

        backtester = VectorizedBacktest(fixed_cost_bps=10.0)
        result = backtester.run(signals, returns, adv)

        assert result.net_sharpe <= result.gross_sharpe

    def test_turnover_computed(self, sample_signals):
        """Test that turnover is computed."""
        signals, returns, adv = sample_signals

        backtester = VectorizedBacktest()
        result = backtester.run(signals, returns, adv)

        assert result.mean_turnover >= 0
        assert result.mean_turnover <= 2  # Max 200%

    def test_weights_sum_correctly(self, sample_signals):
        """Test that weights are normalized."""
        signals, returns, adv = sample_signals

        backtester = VectorizedBacktest()
        result = backtester.run(signals, returns, adv)

        # Check that long weights sum to ~1
        for dt in result.weights.index:
            day_weights = result.weights.loc[dt].dropna()
            long_sum = day_weights[day_weights > 0].sum()
            short_sum = abs(day_weights[day_weights < 0].sum())

            if long_sum > 0:
                assert abs(long_sum - 1) < 0.1
            if short_sum > 0:
                assert abs(short_sum - 1) < 0.1

    def test_ic_metrics(self, sample_signals):
        """Test that IC metrics are computed."""
        signals, returns, adv = sample_signals

        backtester = VectorizedBacktest()
        result = backtester.run(signals, returns, adv)

        assert np.isfinite(result.mean_ic)
        assert np.isfinite(result.icir)
