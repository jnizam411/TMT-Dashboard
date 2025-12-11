"""
Triple Barrier Labeling Method.

Implementation of the triple barrier method from Marcos López de Prado's
"Advances in Financial Machine Learning".

The method defines three barriers:
1. Upper (profit-taking): touch triggers +1 label
2. Lower (stop-loss): touch triggers -1 label
3. Horizontal (time): max holding period

The label is determined by which barrier is touched first.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import structlog
from numba import jit, prange

from ..config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class BarrierEvent:
    """Represents a single barrier touch event."""

    t0: pd.Timestamp  # Entry timestamp
    t1: pd.Timestamp  # Exit timestamp (barrier touch or expiry)
    barrier_side: int  # +1 upper, -1 lower, 0 time barrier
    return_at_touch: float  # Return when barrier was touched
    days_held: int  # Number of days held


def compute_daily_volatility(
    close: pd.Series,
    lookback: int = 20,
    span: Optional[int] = None,
) -> pd.Series:
    """
    Compute daily volatility (standard deviation of returns).

    Args:
        close: Close price series
        lookback: Lookback window for simple rolling std
        span: EWM span (if provided, uses exponential weighting)

    Returns:
        Daily volatility series
    """
    returns = close.pct_change()

    if span is not None:
        vol = returns.ewm(span=span).std()
    else:
        vol = returns.rolling(lookback).std()

    return vol


@jit(nopython=True)
def _find_first_touch_numba(
    prices: np.ndarray,
    entry_idx: int,
    upper_barrier: float,
    lower_barrier: float,
    max_days: int,
) -> Tuple[int, int, float]:
    """
    Numba-accelerated barrier touch detection.

    Returns:
        (exit_idx, barrier_side, return_at_touch)
    """
    entry_price = prices[entry_idx]
    n = len(prices)

    for i in range(1, max_days + 1):
        idx = entry_idx + i
        if idx >= n:
            # Reached end of data
            actual_return = (prices[n - 1] - entry_price) / entry_price
            return n - 1, 0, actual_return

        current_price = prices[idx]
        current_return = (current_price - entry_price) / entry_price

        # Check upper barrier (profit taking)
        if current_return >= upper_barrier:
            return idx, 1, current_return

        # Check lower barrier (stop loss)
        if current_return <= -lower_barrier:
            return idx, -1, current_return

    # Time barrier hit
    exit_idx = min(entry_idx + max_days, n - 1)
    final_return = (prices[exit_idx] - entry_price) / entry_price
    return exit_idx, 0, final_return


class TripleBarrierLabeler:
    """
    Triple barrier labeling for financial time series.

    Parameters are battle-tested defaults from professional quant systems.
    """

    def __init__(
        self,
        horizontal_barrier_days: int = 10,
        vertical_barrier_multiplier: float = 1.5,
        vol_lookback: int = 20,
        min_return_threshold: float = 0.005,
        settings=None,
    ):
        """
        Initialize triple barrier labeler.

        Args:
            horizontal_barrier_days: Maximum holding period in trading days
            vertical_barrier_multiplier: Multiplier on volatility for barriers
            vol_lookback: Lookback for volatility calculation
            min_return_threshold: Minimum return for non-zero label (50 bps default)
        """
        self.settings = settings or get_settings()

        self.horizontal_barrier_days = (
            horizontal_barrier_days or
            self.settings.labeling.horizontal_barrier_days
        )
        self.vertical_barrier_multiplier = (
            vertical_barrier_multiplier or
            self.settings.labeling.vertical_barrier_multiplier
        )
        self.vol_lookback = (
            vol_lookback or
            self.settings.labeling.vol_lookback
        )
        self.min_return_threshold = (
            min_return_threshold or
            self.settings.labeling.min_return_threshold
        )

    def get_barrier_events(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        vol_col: Optional[str] = None,
        entry_dates: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """
        Generate barrier events for all dates in the DataFrame.

        Args:
            df: DataFrame with price data (DatetimeIndex)
            price_col: Column name for price
            vol_col: Column name for pre-computed volatility (optional)
            entry_dates: Specific dates to compute labels (default: all dates)

        Returns:
            DataFrame with columns:
            - t0: Entry timestamp
            - t1: Exit timestamp
            - first_touch_date: Same as t1
            - barrier_side: +1 (upper), -1 (lower), 0 (time)
            - return_at_touch: Return when barrier touched
            - days_held: Days between entry and exit
            - label: Final label (+1, 0, -1)
        """
        df = df.copy()

        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Compute volatility if not provided
        if vol_col is None or vol_col not in df.columns:
            df["volatility"] = compute_daily_volatility(
                df[price_col],
                lookback=self.vol_lookback,
            )
            vol_col = "volatility"

        # Use all dates if not specified
        if entry_dates is None:
            # Skip first vol_lookback days (no volatility estimate)
            entry_dates = df.index[self.vol_lookback:]

        # Convert to numpy for numba
        prices = df[price_col].values
        volatilities = df[vol_col].values
        dates = df.index

        # Create index mapping
        date_to_idx = {d: i for i, d in enumerate(dates)}

        events = []

        for entry_date in entry_dates:
            if entry_date not in date_to_idx:
                continue

            entry_idx = date_to_idx[entry_date]

            # Get volatility at entry
            vol = volatilities[entry_idx]
            if np.isnan(vol) or vol <= 0:
                continue

            # Compute barriers
            upper_barrier = self.vertical_barrier_multiplier * vol
            lower_barrier = self.vertical_barrier_multiplier * vol

            # Find first touch
            exit_idx, barrier_side, return_at_touch = _find_first_touch_numba(
                prices,
                entry_idx,
                upper_barrier,
                lower_barrier,
                self.horizontal_barrier_days,
            )

            # Determine final label
            if barrier_side == 0:
                # Time barrier - check if return exceeds minimum threshold
                if abs(return_at_touch) < self.min_return_threshold:
                    label = 0
                else:
                    label = 1 if return_at_touch > 0 else -1
            else:
                label = barrier_side

            events.append({
                "t0": entry_date,
                "t1": dates[exit_idx],
                "first_touch_date": dates[exit_idx],
                "barrier_side": barrier_side,
                "return_at_touch": return_at_touch,
                "days_held": exit_idx - entry_idx,
                "label": label,
                "upper_barrier": upper_barrier,
                "lower_barrier": lower_barrier,
            })

        events_df = pd.DataFrame(events)

        if len(events_df) > 0:
            events_df = events_df.set_index("t0")
            events_df.index.name = "t0"

        logger.info(f"Generated {len(events_df)} barrier events",
                   label_dist=events_df["label"].value_counts().to_dict() if len(events_df) > 0 else {})

        return events_df

    def get_sample_weights(
        self,
        events: pd.DataFrame,
        close: pd.Series,
        num_threads: int = 1,
    ) -> pd.Series:
        """
        Compute sample weights based on uniqueness of labels.

        Returns weights that account for overlapping labels,
        giving less weight to samples that overlap more.
        """
        # Concurrent labels at each timestamp
        t1 = events["t1"]

        # For each timestamp, count how many events span it
        concurrent = pd.Series(0, index=close.index)

        for t0, row in events.iterrows():
            mask = (close.index >= t0) & (close.index <= row["t1"])
            concurrent.loc[mask] += 1

        # Average uniqueness for each event
        weights = pd.Series(index=events.index, dtype=float)

        for t0, row in events.iterrows():
            mask = (close.index >= t0) & (close.index <= row["t1"])
            avg_concurrent = concurrent.loc[mask].mean()
            weights.loc[t0] = 1.0 / avg_concurrent if avg_concurrent > 0 else 1.0

        # Normalize to sum to 1
        weights = weights / weights.sum()

        return weights


def get_barrier_events(
    df: pd.DataFrame,
    horizontal_barrier_days: int = 10,
    vertical_barrier_multiplier: float = 1.5,
    vol_lookback: int = 20,
    min_return_threshold: float = 0.005,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Convenience function to get barrier events.

    Args:
        df: DataFrame with price data
        horizontal_barrier_days: Max holding period
        vertical_barrier_multiplier: Vol multiplier for barriers
        vol_lookback: Lookback for volatility
        min_return_threshold: Min return for non-zero label
        price_col: Price column name

    Returns:
        DataFrame with barrier events
    """
    labeler = TripleBarrierLabeler(
        horizontal_barrier_days=horizontal_barrier_days,
        vertical_barrier_multiplier=vertical_barrier_multiplier,
        vol_lookback=vol_lookback,
        min_return_threshold=min_return_threshold,
    )
    return labeler.get_barrier_events(df, price_col=price_col)


@jit(nopython=True, parallel=True)
def _batch_find_touches_numba(
    prices: np.ndarray,
    entry_indices: np.ndarray,
    upper_barriers: np.ndarray,
    lower_barriers: np.ndarray,
    max_days: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch process multiple entries in parallel using Numba.

    Returns:
        (exit_indices, barrier_sides, returns)
    """
    n_entries = len(entry_indices)
    exit_indices = np.empty(n_entries, dtype=np.int64)
    barrier_sides = np.empty(n_entries, dtype=np.int64)
    returns = np.empty(n_entries, dtype=np.float64)

    for i in prange(n_entries):
        exit_idx, side, ret = _find_first_touch_numba(
            prices,
            entry_indices[i],
            upper_barriers[i],
            lower_barriers[i],
            max_days,
        )
        exit_indices[i] = exit_idx
        barrier_sides[i] = side
        returns[i] = ret

    return exit_indices, barrier_sides, returns


class BatchTripleBarrierLabeler(TripleBarrierLabeler):
    """
    Optimized batch processing for large datasets.

    Uses parallel Numba for significant speedup on large universes.
    """

    def get_barrier_events_batch(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        vol_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Batch process all valid dates using parallel Numba.
        """
        df = df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Compute volatility
        if vol_col is None or vol_col not in df.columns:
            df["volatility"] = compute_daily_volatility(
                df[price_col],
                lookback=self.vol_lookback,
            )
            vol_col = "volatility"

        # Prepare arrays
        prices = df[price_col].values
        volatilities = df[vol_col].values
        dates = df.index

        # Valid entries (have volatility)
        valid_mask = ~np.isnan(volatilities) & (volatilities > 0)
        valid_indices = np.where(valid_mask)[0]

        # Skip entries too close to end
        max_idx = len(prices) - self.horizontal_barrier_days - 1
        valid_indices = valid_indices[valid_indices < max_idx]

        if len(valid_indices) == 0:
            return pd.DataFrame()

        # Compute barriers for all entries
        upper_barriers = self.vertical_barrier_multiplier * volatilities[valid_indices]
        lower_barriers = self.vertical_barrier_multiplier * volatilities[valid_indices]

        # Batch process
        exit_indices, barrier_sides, returns = _batch_find_touches_numba(
            prices,
            valid_indices.astype(np.int64),
            upper_barriers,
            lower_barriers,
            self.horizontal_barrier_days,
        )

        # Build DataFrame
        events = []
        for i, entry_idx in enumerate(valid_indices):
            return_at_touch = returns[i]
            barrier_side = barrier_sides[i]

            if barrier_side == 0:
                if abs(return_at_touch) < self.min_return_threshold:
                    label = 0
                else:
                    label = 1 if return_at_touch > 0 else -1
            else:
                label = barrier_side

            events.append({
                "t0": dates[entry_idx],
                "t1": dates[int(exit_indices[i])],
                "first_touch_date": dates[int(exit_indices[i])],
                "barrier_side": int(barrier_side),
                "return_at_touch": return_at_touch,
                "days_held": int(exit_indices[i] - entry_idx),
                "label": label,
            })

        events_df = pd.DataFrame(events)
        if len(events_df) > 0:
            events_df = events_df.set_index("t0")

        return events_df
