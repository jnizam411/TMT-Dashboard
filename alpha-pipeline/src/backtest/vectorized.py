"""
Vectorized Backtesting Engine.

High-performance backtesting with:
- Realistic transaction costs
- Position limits
- Sector neutrality
- Daily rebalancing
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

from ..config import get_settings
from ..validation.metrics import (
    compute_sharpe,
    compute_max_drawdown,
    compute_turnover,
    compute_daily_ic,
)
from .costs import TransactionCostModel, MarketImpactModel

logger = structlog.get_logger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""

    # Returns
    gross_returns: pd.Series
    net_returns: pd.Series
    cumulative_returns: pd.Series

    # Metrics
    gross_sharpe: float
    net_sharpe: float
    max_drawdown: float
    mean_turnover: float
    total_return: float
    annualized_return: float
    volatility: float
    calmar_ratio: float

    # IC metrics
    mean_ic: float
    icir: float

    # Costs
    total_costs: float
    avg_daily_cost: float

    # Positions
    weights: pd.DataFrame
    positions: pd.DataFrame

    # Additional data
    daily_pnl: pd.Series = field(default_factory=pd.Series)
    sector_exposures: pd.DataFrame = field(default_factory=pd.DataFrame)


class VectorizedBacktest:
    """
    Vectorized backtesting engine.

    Implements efficient portfolio simulation with realistic
    constraints and costs.
    """

    def __init__(
        self,
        fixed_cost_bps: float = 7.5,
        impact_bps_per_mm: float = 0.1,
        max_position_adv_pct: float = 0.05,
        max_sector_weight: float = 0.25,
        sector_neutral: bool = True,
        initial_capital: float = 10_000_000,
        rebalance_frequency: str = "daily",
        settings=None,
    ):
        """
        Initialize backtester.

        Args:
            fixed_cost_bps: Fixed transaction cost in bps
            impact_bps_per_mm: Market impact per $1M
            max_position_adv_pct: Max position as % of ADV
            max_sector_weight: Max sector weight
            sector_neutral: Enforce sector neutrality
            initial_capital: Starting capital
            rebalance_frequency: "daily", "weekly", "monthly"
        """
        self.settings = settings or get_settings()

        self.fixed_cost_bps = fixed_cost_bps or self.settings.backtest.fixed_cost_bps
        self.impact_bps_per_mm = impact_bps_per_mm or self.settings.backtest.impact_bps_per_mm
        self.max_position_adv_pct = max_position_adv_pct or self.settings.backtest.max_position_adv_pct
        self.max_sector_weight = max_sector_weight or self.settings.backtest.max_sector_weight
        self.sector_neutral = sector_neutral
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency

        # Cost models
        self.cost_model = TransactionCostModel(fixed_cost_bps=fixed_cost_bps)
        self.impact_model = MarketImpactModel(impact_bps_per_mm=impact_bps_per_mm)

    def run(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        adv: Optional[pd.DataFrame] = None,
        sectors: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """
        Run vectorized backtest.

        Args:
            signals: DataFrame with prediction scores (dates x tickers)
            returns: DataFrame with forward returns (dates x tickers)
            adv: DataFrame with average daily volume (dates x tickers)
            sectors: Series mapping ticker to sector

        Returns:
            BacktestResult with all metrics
        """
        # Align data
        common_dates = signals.index.intersection(returns.index)
        common_tickers = signals.columns.intersection(returns.columns)

        signals = signals.loc[common_dates, common_tickers]
        returns = returns.loc[common_dates, common_tickers]

        if adv is not None:
            adv = adv.loc[common_dates, common_tickers.intersection(adv.columns)]

        logger.info(f"Running backtest on {len(common_dates)} days, "
                   f"{len(common_tickers)} tickers")

        # Generate target weights
        target_weights = self._compute_target_weights(signals, sectors)

        # Apply position limits
        if adv is not None:
            target_weights = self._apply_position_limits(target_weights, adv)

        # Apply sector constraints
        if self.sector_neutral and sectors is not None:
            target_weights = self._apply_sector_neutrality(target_weights, sectors)

        # Normalize weights
        target_weights = self._normalize_weights(target_weights)

        # Compute portfolio returns and costs
        gross_returns, net_returns, costs = self._compute_returns(
            target_weights, returns, adv
        )

        # Compute metrics
        result = self._compute_metrics(
            gross_returns,
            net_returns,
            target_weights,
            signals,
            returns,
            costs,
        )

        return result

    def _compute_target_weights(
        self,
        signals: pd.DataFrame,
        sectors: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Convert signals to target portfolio weights.

        Long/short based on top/bottom decile.
        """
        n_quantiles = 10
        weights = pd.DataFrame(
            np.nan,
            index=signals.index,
            columns=signals.columns,
        )

        for dt in signals.index:
            day_signals = signals.loc[dt].dropna()

            if len(day_signals) < n_quantiles:
                continue

            # Rank stocks
            ranks = day_signals.rank(pct=True)

            # Long top decile, short bottom decile
            long_mask = ranks >= 0.9
            short_mask = ranks <= 0.1

            n_long = long_mask.sum()
            n_short = short_mask.sum()

            day_weights = pd.Series(0.0, index=day_signals.index)

            if n_long > 0:
                day_weights[long_mask] = 1.0 / n_long

            if n_short > 0:
                day_weights[short_mask] = -1.0 / n_short

            weights.loc[dt, day_weights.index] = day_weights

        return weights

    def _apply_position_limits(
        self,
        weights: pd.DataFrame,
        adv: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply position limits based on ADV."""
        # Compute max weight based on ADV
        max_weight = self.max_position_adv_pct

        for dt in weights.index:
            if dt not in adv.index:
                continue

            day_weights = weights.loc[dt]
            day_adv = adv.loc[dt]

            # Scale down positions exceeding limit
            for ticker in day_weights.index:
                if ticker not in day_adv.index:
                    continue

                ticker_adv = day_adv[ticker]
                current_weight = abs(day_weights[ticker])

                # Estimate position size in dollars
                position_size = current_weight * self.initial_capital
                adv_limit = ticker_adv * self.max_position_adv_pct

                if position_size > adv_limit and ticker_adv > 0:
                    scale = adv_limit / position_size
                    weights.loc[dt, ticker] *= scale

        return weights

    def _apply_sector_neutrality(
        self,
        weights: pd.DataFrame,
        sectors: pd.Series,
    ) -> pd.DataFrame:
        """Apply sector neutrality constraint."""
        for dt in weights.index:
            day_weights = weights.loc[dt].dropna()

            if len(day_weights) == 0:
                continue

            # Get sectors for current holdings
            tickers_with_sectors = day_weights.index.intersection(sectors.index)
            if len(tickers_with_sectors) == 0:
                continue

            # Compute sector exposures
            sector_weights = pd.Series(0.0, index=sectors.unique())

            for ticker in tickers_with_sectors:
                sector = sectors[ticker]
                sector_weights[sector] += day_weights[ticker]

            # Adjust to neutralize sectors
            for sector in sector_weights.index:
                sector_exposure = sector_weights[sector]

                if abs(sector_exposure) > 0.01:  # Non-trivial exposure
                    # Get tickers in this sector
                    sector_tickers = sectors[sectors == sector].index
                    sector_tickers = sector_tickers.intersection(tickers_with_sectors)

                    if len(sector_tickers) > 0:
                        # Reduce all positions in sector proportionally
                        adjustment = sector_exposure / len(sector_tickers)
                        for ticker in sector_tickers:
                            weights.loc[dt, ticker] -= adjustment

        return weights

    def _normalize_weights(
        self,
        weights: pd.DataFrame,
    ) -> pd.DataFrame:
        """Normalize weights to sum to target leverage (1.0 long, 1.0 short)."""
        for dt in weights.index:
            day_weights = weights.loc[dt].dropna()

            long_sum = day_weights[day_weights > 0].sum()
            short_sum = abs(day_weights[day_weights < 0].sum())

            if long_sum > 0:
                weights.loc[dt, day_weights > 0] /= long_sum

            if short_sum > 0:
                weights.loc[dt, day_weights < 0] /= short_sum

        return weights

    def _compute_returns(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        adv: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute gross and net returns.

        Returns:
            (gross_returns, net_returns, costs)
        """
        # Gross returns = sum(weight * return)
        gross_returns = (weights.shift(1) * returns).sum(axis=1)

        # Compute turnover and costs
        weight_changes = weights.diff().abs()
        daily_turnover = weight_changes.sum(axis=1) / 2

        # Fixed cost
        fixed_cost = daily_turnover * self.fixed_cost_bps / 10000

        # Market impact cost
        impact_cost = pd.Series(0.0, index=weights.index)

        if adv is not None:
            for dt in weights.index:
                if dt not in adv.index:
                    continue

                day_changes = weight_changes.loc[dt].dropna()
                day_adv = adv.loc[dt]

                total_impact = 0
                for ticker in day_changes.index:
                    if ticker not in day_adv.index:
                        continue

                    trade_value = abs(day_changes[ticker]) * self.initial_capital
                    ticker_adv = day_adv[ticker] * (day_adv[ticker] > 0).astype(float) + 1e-10

                    impact = self.impact_model.compute_impact(trade_value, ticker_adv)
                    total_impact += impact

                impact_cost[dt] = total_impact / self.initial_capital

        total_cost = fixed_cost + impact_cost
        net_returns = gross_returns - total_cost

        return gross_returns, net_returns, total_cost

    def _compute_metrics(
        self,
        gross_returns: pd.Series,
        net_returns: pd.Series,
        weights: pd.DataFrame,
        signals: pd.DataFrame,
        forward_returns: pd.DataFrame,
        costs: pd.Series,
    ) -> BacktestResult:
        """Compute all backtest metrics."""
        # Clean returns
        gross_returns = gross_returns.dropna()
        net_returns = net_returns.dropna()

        # Cumulative returns
        cumulative = (1 + net_returns).cumprod()

        # Sharpe ratios
        gross_sharpe = compute_sharpe(gross_returns)
        net_sharpe = compute_sharpe(net_returns)

        # Max drawdown
        max_dd = compute_max_drawdown(net_returns)

        # Turnover
        daily_turnover = compute_turnover(weights)
        mean_turnover = daily_turnover.mean()

        # Total return and annualized
        total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
        n_years = len(net_returns) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Volatility
        volatility = net_returns.std() * np.sqrt(252)

        # Calmar ratio
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0

        # IC metrics
        daily_ic = compute_daily_ic(signals, forward_returns)
        mean_ic = daily_ic.mean()
        icir = mean_ic / daily_ic.std() * np.sqrt(252) if daily_ic.std() > 0 else 0

        # Cost summary
        total_costs = costs.sum()
        avg_daily_cost = costs.mean()

        return BacktestResult(
            gross_returns=gross_returns,
            net_returns=net_returns,
            cumulative_returns=cumulative,
            gross_sharpe=gross_sharpe,
            net_sharpe=net_sharpe,
            max_drawdown=max_dd,
            mean_turnover=mean_turnover,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            calmar_ratio=calmar,
            mean_ic=mean_ic,
            icir=icir,
            total_costs=total_costs,
            avg_daily_cost=avg_daily_cost,
            weights=weights,
            positions=weights * self.initial_capital,
        )

    def run_walk_forward(
        self,
        signals_by_split: Dict[int, pd.DataFrame],
        returns: pd.DataFrame,
        adv: Optional[pd.DataFrame] = None,
        sectors: Optional[pd.Series] = None,
    ) -> Dict[int, BacktestResult]:
        """
        Run walk-forward backtest across multiple splits.

        Returns:
            Dict of split_id -> BacktestResult
        """
        results = {}

        for split_id, signals in signals_by_split.items():
            logger.info(f"Running backtest for split {split_id}")

            result = self.run(signals, returns, adv, sectors)
            results[split_id] = result

        # Aggregate metrics
        self._log_aggregate_metrics(results)

        return results

    def _log_aggregate_metrics(
        self,
        results: Dict[int, BacktestResult],
    ):
        """Log aggregate metrics across splits."""
        sharpes = [r.net_sharpe for r in results.values()]
        ics = [r.mean_ic for r in results.values()]
        turnovers = [r.mean_turnover for r in results.values()]
        drawdowns = [r.max_drawdown for r in results.values()]

        logger.info("Aggregate Walk-Forward Results:")
        logger.info(f"  Mean Net Sharpe: {np.mean(sharpes):.3f} "
                   f"(std: {np.std(sharpes):.3f})")
        logger.info(f"  Mean IC: {np.mean(ics):.4f}")
        logger.info(f"  Mean Turnover: {np.mean(turnovers):.2%}")
        logger.info(f"  Mean Max DD: {np.mean(drawdowns):.2%}")


class PortfolioAnalyzer:
    """
    Portfolio analysis utilities.
    """

    @staticmethod
    def compute_sector_exposures(
        weights: pd.DataFrame,
        sectors: pd.Series,
    ) -> pd.DataFrame:
        """Compute daily sector exposures."""
        exposures = {}

        for dt in weights.index:
            day_weights = weights.loc[dt].dropna()
            day_exposure = {}

            for ticker in day_weights.index:
                if ticker in sectors.index:
                    sector = sectors[ticker]
                    day_exposure[sector] = day_exposure.get(sector, 0) + day_weights[ticker]

            exposures[dt] = day_exposure

        return pd.DataFrame(exposures).T.fillna(0)

    @staticmethod
    def compute_factor_exposures(
        weights: pd.DataFrame,
        factor_loadings: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute daily factor exposures."""
        exposures = {}

        for dt in weights.index:
            day_weights = weights.loc[dt].dropna()

            if dt not in factor_loadings.index:
                continue

            day_loadings = factor_loadings.loc[dt]
            common = day_weights.index.intersection(day_loadings.columns)

            if len(common) == 0:
                continue

            exposure = (day_weights[common] * day_loadings[common]).sum()
            exposures[dt] = exposure

        return pd.DataFrame(exposures).T

    @staticmethod
    def compute_concentration(
        weights: pd.DataFrame,
    ) -> pd.Series:
        """Compute portfolio concentration (Herfindahl index)."""
        concentration = (weights ** 2).sum(axis=1)
        return concentration
