"""
Transaction Cost Models.

Realistic cost modeling including:
- Fixed costs (commissions, fees)
- Market impact (square-root law)
- Bid-ask spread
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import structlog

from ..config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class TransactionCost:
    """Container for transaction cost components."""

    fixed_cost: float  # Fixed cost in dollars
    spread_cost: float  # Bid-ask spread cost
    impact_cost: float  # Market impact cost
    total_cost: float  # Total cost


class TransactionCostModel:
    """
    Combined transaction cost model.

    Components:
    1. Fixed costs: commissions, regulatory fees
    2. Spread costs: half the bid-ask spread
    3. Market impact: price movement from trading
    """

    def __init__(
        self,
        fixed_cost_bps: float = 7.5,
        spread_cost_bps: float = 2.0,
        settings=None,
    ):
        """
        Initialize cost model.

        Args:
            fixed_cost_bps: Fixed cost in basis points
            spread_cost_bps: Spread cost in basis points
        """
        self.settings = settings or get_settings()
        self.fixed_cost_bps = fixed_cost_bps or self.settings.backtest.fixed_cost_bps
        self.spread_cost_bps = spread_cost_bps

    def compute_cost(
        self,
        trade_value: float,
        adv: Optional[float] = None,
    ) -> TransactionCost:
        """
        Compute transaction cost for a single trade.

        Args:
            trade_value: Absolute value of trade in dollars
            adv: Average daily volume in dollars (for impact)

        Returns:
            TransactionCost breakdown
        """
        fixed_cost = trade_value * self.fixed_cost_bps / 10000
        spread_cost = trade_value * self.spread_cost_bps / 10000

        # Impact cost (if ADV available)
        if adv is not None and adv > 0:
            impact_model = MarketImpactModel()
            impact_cost = impact_model.compute_impact(trade_value, adv)
        else:
            impact_cost = 0

        total_cost = fixed_cost + spread_cost + impact_cost

        return TransactionCost(
            fixed_cost=fixed_cost,
            spread_cost=spread_cost,
            impact_cost=impact_cost,
            total_cost=total_cost,
        )

    def compute_portfolio_costs(
        self,
        trades: pd.DataFrame,
        adv: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Compute costs for a portfolio of trades.

        Args:
            trades: DataFrame with columns [ticker, trade_value]
            adv: Series of ADV by ticker

        Returns:
            DataFrame with cost breakdown by ticker
        """
        results = []

        for _, row in trades.iterrows():
            ticker = row.get("ticker", row.name)
            trade_value = abs(row["trade_value"])

            ticker_adv = adv.get(ticker) if adv is not None else None

            cost = self.compute_cost(trade_value, ticker_adv)

            results.append({
                "ticker": ticker,
                "trade_value": trade_value,
                "fixed_cost": cost.fixed_cost,
                "spread_cost": cost.spread_cost,
                "impact_cost": cost.impact_cost,
                "total_cost": cost.total_cost,
            })

        return pd.DataFrame(results)


class MarketImpactModel:
    """
    Market impact model using square-root law.

    Impact = impact_bps_per_mm * sqrt(trade_value / 1_000_000)

    This is the standard Almgren-Chriss style temporary impact model.
    """

    def __init__(
        self,
        impact_bps_per_mm: float = 0.1,
        permanent_impact_ratio: float = 0.2,
        settings=None,
    ):
        """
        Initialize market impact model.

        Args:
            impact_bps_per_mm: Basis points impact per $1M notional
            permanent_impact_ratio: Ratio of permanent to temporary impact
        """
        self.settings = settings or get_settings()
        self.impact_bps_per_mm = (
            impact_bps_per_mm or
            self.settings.backtest.impact_bps_per_mm
        )
        self.permanent_impact_ratio = permanent_impact_ratio

    def compute_impact(
        self,
        trade_value: float,
        adv: float,
    ) -> float:
        """
        Compute market impact cost.

        Uses square-root law: impact proportional to sqrt(trade_size / ADV)

        Args:
            trade_value: Trade value in dollars
            adv: Average daily volume in dollars

        Returns:
            Impact cost in dollars
        """
        if adv <= 0:
            return 0

        # Participation rate
        participation = trade_value / adv

        # Square root impact
        impact_bps = self.impact_bps_per_mm * np.sqrt(trade_value / 1_000_000)

        # Scale by participation rate
        impact_bps *= np.sqrt(participation)

        impact_cost = trade_value * impact_bps / 10000

        return impact_cost

    def compute_temporary_impact(
        self,
        trade_value: float,
        adv: float,
    ) -> float:
        """
        Compute temporary (transient) market impact.

        This is the cost that affects only the current trade.
        """
        total_impact = self.compute_impact(trade_value, adv)
        return total_impact * (1 - self.permanent_impact_ratio)

    def compute_permanent_impact(
        self,
        trade_value: float,
        adv: float,
    ) -> float:
        """
        Compute permanent market impact.

        This is the cost that persists after the trade.
        """
        total_impact = self.compute_impact(trade_value, adv)
        return total_impact * self.permanent_impact_ratio

    def estimate_optimal_trade_size(
        self,
        total_order: float,
        adv: float,
        risk_aversion: float = 0.01,
        n_periods: int = 1,
    ) -> float:
        """
        Estimate optimal trade size per period.

        Based on Almgren-Chriss optimal execution.

        Args:
            total_order: Total order size in dollars
            adv: Average daily volume
            risk_aversion: Risk aversion parameter
            n_periods: Number of trading periods

        Returns:
            Optimal trade size per period
        """
        # Simple TWAP for now
        return total_order / n_periods


class SlippageModel:
    """
    Slippage model for realistic execution simulation.
    """

    def __init__(
        self,
        base_slippage_bps: float = 2.0,
        volatility_multiplier: float = 0.5,
    ):
        """
        Initialize slippage model.

        Args:
            base_slippage_bps: Base slippage in basis points
            volatility_multiplier: Multiplier for volatility-based slippage
        """
        self.base_slippage_bps = base_slippage_bps
        self.volatility_multiplier = volatility_multiplier

    def compute_slippage(
        self,
        trade_value: float,
        daily_volatility: Optional[float] = None,
    ) -> float:
        """
        Compute execution slippage.

        Args:
            trade_value: Trade value in dollars
            daily_volatility: Daily volatility (optional)

        Returns:
            Slippage cost in dollars
        """
        base_cost = trade_value * self.base_slippage_bps / 10000

        if daily_volatility is not None:
            vol_cost = trade_value * daily_volatility * self.volatility_multiplier
            return base_cost + vol_cost

        return base_cost


class BorrowCostModel:
    """
    Stock borrow cost model for short positions.
    """

    def __init__(
        self,
        general_collateral_rate: float = 0.003,  # 30 bps annual
        hard_to_borrow_rate: float = 0.10,  # 10% annual
    ):
        """
        Initialize borrow cost model.

        Args:
            general_collateral_rate: Annual rate for easy-to-borrow
            hard_to_borrow_rate: Annual rate for hard-to-borrow
        """
        self.gc_rate = general_collateral_rate
        self.htb_rate = hard_to_borrow_rate

    def compute_daily_borrow_cost(
        self,
        short_value: float,
        is_hard_to_borrow: bool = False,
    ) -> float:
        """
        Compute daily borrow cost for short position.

        Args:
            short_value: Absolute value of short position
            is_hard_to_borrow: Whether stock is hard to borrow

        Returns:
            Daily borrow cost in dollars
        """
        annual_rate = self.htb_rate if is_hard_to_borrow else self.gc_rate
        daily_cost = short_value * annual_rate / 252

        return daily_cost
