"""
Purged Walk-Forward Validation.

The most rigorous validation framework for financial time series.
Implements proper purging to prevent information leakage.

Key parameters (battle-tested):
- 63-day purge gap between train and validation
- 252-day validation and test periods
- Rolling forward by 252 days each iteration

This gives approximately 10 completely independent out-of-sample periods.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

from ..config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class SplitInfo:
    """Information about a single train/val/test split."""

    split_id: int
    train_start: date
    train_end: date
    purge_end: date
    val_start: date
    val_end: date
    test_start: date
    test_end: date
    train_tickers: List[str]
    val_tickers: List[str]
    test_tickers: List[str]


class PurgedWalkForwardSplitter:
    """
    Purged Walk-Forward Cross-Validation for financial time series.

    Implementation of the proper validation framework from
    Marcos López de Prado's work, with 63-day purge gaps.

    Default configuration yields ~10 independent out-of-sample periods
    from 2005-2025.
    """

    def __init__(
        self,
        train_start: Optional[date] = None,
        train_end: Optional[date] = None,
        purge_days: int = 63,
        val_days: int = 252,
        test_days: int = 252,
        roll_days: int = 252,
        final_test_end: Optional[date] = None,
        settings=None,
    ):
        """
        Initialize walk-forward splitter.

        Args:
            train_start: Start of initial training period
            train_end: End of initial training period
            purge_days: Trading days to purge between train/val
            val_days: Trading days in validation period
            test_days: Trading days in test period
            roll_days: Trading days to roll forward each iteration
            final_test_end: End date for final test period
        """
        self.settings = settings or get_settings()

        self.train_start = train_start or self.settings.validation.train_start
        self.train_end = train_end or self.settings.validation.train_end
        self.purge_days = purge_days or self.settings.validation.purge_days
        self.val_days = val_days or self.settings.validation.val_days
        self.test_days = test_days or self.settings.validation.test_days
        self.roll_days = roll_days or self.settings.validation.roll_days
        self.final_test_end = final_test_end or date(2025, 12, 31)

        self._splits = None
        self._universe_df = None

    def _trading_days_forward(
        self,
        start_date: date,
        n_days: int,
        trading_days: pd.DatetimeIndex,
    ) -> date:
        """Get date n trading days forward from start."""
        start_ts = pd.Timestamp(start_date)
        future_days = trading_days[trading_days > start_ts]

        if len(future_days) < n_days:
            return trading_days[-1].date()

        return future_days[n_days - 1].date()

    def _trading_days_between(
        self,
        start_date: date,
        end_date: date,
        trading_days: pd.DatetimeIndex,
    ) -> int:
        """Count trading days between two dates."""
        mask = (trading_days >= pd.Timestamp(start_date)) & \
               (trading_days <= pd.Timestamp(end_date))
        return mask.sum()

    def set_universe(self, universe_df: pd.DataFrame):
        """
        Set the universe DataFrame for ticker lookups.

        Args:
            universe_df: DataFrame with columns [date, ticker, in_universe]
        """
        self._universe_df = universe_df

    def _get_tickers_for_period(
        self,
        start_date: date,
        end_date: date,
    ) -> List[str]:
        """Get list of tickers available during a period."""
        if self._universe_df is None:
            return []

        mask = (
            (self._universe_df["date"].dt.date >= start_date) &
            (self._universe_df["date"].dt.date <= end_date) &
            (self._universe_df["in_universe"] == True)
        )

        return self._universe_df.loc[mask, "ticker"].unique().tolist()

    def generate_splits(
        self,
        trading_days: Optional[pd.DatetimeIndex] = None,
    ) -> List[SplitInfo]:
        """
        Generate all train/val/test splits.

        Args:
            trading_days: Trading calendar (if None, uses business days)

        Returns:
            List of SplitInfo objects describing each split
        """
        if trading_days is None:
            trading_days = pd.bdate_range(self.train_start, self.final_test_end)

        splits = []
        split_id = 0
        current_train_end = self.train_end

        while True:
            # Purge period
            purge_end = self._trading_days_forward(
                current_train_end, self.purge_days, trading_days
            )

            # Validation period
            val_start = self._trading_days_forward(
                purge_end, 1, trading_days
            )
            val_end = self._trading_days_forward(
                val_start, self.val_days, trading_days
            )

            # Test period
            test_start = self._trading_days_forward(
                val_end, 1, trading_days
            )
            test_end = self._trading_days_forward(
                test_start, self.test_days, trading_days
            )

            # Check if we've gone past the final date
            if test_end > self.final_test_end:
                test_end = self.final_test_end
                # If test period is too short, stop
                test_trading_days = self._trading_days_between(
                    test_start, test_end, trading_days
                )
                if test_trading_days < 63:  # Minimum 3 months
                    break

            # Get tickers for each period
            train_tickers = self._get_tickers_for_period(
                self.train_start, current_train_end
            )
            val_tickers = self._get_tickers_for_period(val_start, val_end)
            test_tickers = self._get_tickers_for_period(test_start, test_end)

            split = SplitInfo(
                split_id=split_id,
                train_start=self.train_start,
                train_end=current_train_end,
                purge_end=purge_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
                train_tickers=train_tickers,
                val_tickers=val_tickers,
                test_tickers=test_tickers,
            )

            splits.append(split)
            split_id += 1

            # Roll forward
            current_train_end = self._trading_days_forward(
                current_train_end, self.roll_days, trading_days
            )

            # Stop if we've covered all data
            if test_end >= self.final_test_end:
                break

        self._splits = splits
        logger.info(f"Generated {len(splits)} walk-forward splits")

        return splits

    def get_split(self, split_id: int) -> SplitInfo:
        """Get a specific split by ID."""
        if self._splits is None:
            self._splits = self.generate_splits()

        return self._splits[split_id]

    def n_splits(self) -> int:
        """Get number of splits."""
        if self._splits is None:
            self._splits = self.generate_splits()
        return len(self._splits)

    def get_train_val_test_indices(
        self,
        data_index: pd.DatetimeIndex,
        split_id: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get indices for train/val/test for a given split.

        Args:
            data_index: DatetimeIndex of the data
            split_id: Split ID

        Returns:
            (train_indices, val_indices, test_indices)
        """
        split = self.get_split(split_id)

        train_mask = (data_index >= pd.Timestamp(split.train_start)) & \
                     (data_index <= pd.Timestamp(split.train_end))

        val_mask = (data_index >= pd.Timestamp(split.val_start)) & \
                   (data_index <= pd.Timestamp(split.val_end))

        test_mask = (data_index >= pd.Timestamp(split.test_start)) & \
                    (data_index <= pd.Timestamp(split.test_end))

        return (
            np.where(train_mask)[0],
            np.where(val_mask)[0],
            np.where(test_mask)[0],
        )

    def split_panel_data(
        self,
        panel: pd.DataFrame,
        split_id: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split panel data into train/val/test.

        Args:
            panel: Panel DataFrame with MultiIndex (date, ticker)
            split_id: Split ID

        Returns:
            (train_df, val_df, test_df)
        """
        split = self.get_split(split_id)

        dates = panel.index.get_level_values("date")

        train_mask = (dates >= pd.Timestamp(split.train_start)) & \
                     (dates <= pd.Timestamp(split.train_end))

        val_mask = (dates >= pd.Timestamp(split.val_start)) & \
                   (dates <= pd.Timestamp(split.val_end))

        test_mask = (dates >= pd.Timestamp(split.test_start)) & \
                    (dates <= pd.Timestamp(split.test_end))

        return (
            panel.loc[train_mask],
            panel.loc[val_mask],
            panel.loc[test_mask],
        )

    def __iter__(self) -> Generator[SplitInfo, None, None]:
        """Iterate over all splits."""
        if self._splits is None:
            self._splits = self.generate_splits()

        for split in self._splits:
            yield split

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all splits."""
        if self._splits is None:
            self._splits = self.generate_splits()

        records = []
        for split in self._splits:
            records.append({
                "split_id": split.split_id,
                "train_start": split.train_start,
                "train_end": split.train_end,
                "val_start": split.val_start,
                "val_end": split.val_end,
                "test_start": split.test_start,
                "test_end": split.test_end,
                "n_train_tickers": len(split.train_tickers),
                "n_val_tickers": len(split.val_tickers),
                "n_test_tickers": len(split.test_tickers),
            })

        return pd.DataFrame(records)


class PurgedKFold:
    """
    Purged K-Fold cross-validation.

    Alternative to walk-forward for cases where we want
    more folds but still need purging.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 63,
        embargo_days: int = 21,
    ):
        """
        Initialize purged K-Fold.

        Args:
            n_splits: Number of folds
            purge_days: Days to purge before test set
            embargo_days: Days to embargo after test set
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices with purging.

        Args:
            X: Feature DataFrame with DatetimeIndex
            y: Target (not used, for sklearn compatibility)
            groups: Groups (not used)

        Yields:
            (train_indices, test_indices) tuples
        """
        dates = X.index.get_level_values("date") if isinstance(X.index, pd.MultiIndex) else X.index
        unique_dates = dates.unique().sort_values()
        n_dates = len(unique_dates)

        fold_size = n_dates // self.n_splits

        for fold_idx in range(self.n_splits):
            # Test period
            test_start_idx = fold_idx * fold_size
            test_end_idx = (fold_idx + 1) * fold_size if fold_idx < self.n_splits - 1 else n_dates

            test_start = unique_dates[test_start_idx]
            test_end = unique_dates[test_end_idx - 1]

            # Create masks
            test_mask = (dates >= test_start) & (dates <= test_end)

            # Purge: exclude purge_days before test
            purge_start = unique_dates[max(0, test_start_idx - self.purge_days)]

            # Embargo: exclude embargo_days after test
            embargo_end_idx = min(n_dates - 1, test_end_idx + self.embargo_days)
            embargo_end = unique_dates[embargo_end_idx]

            # Train mask: everything except test, purge, and embargo
            train_mask = ~(
                (dates >= purge_start) & (dates <= embargo_end)
            )

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            yield train_indices, test_indices


class TemporalSplit:
    """
    Simple temporal train/test split with purging.

    For single split scenarios.
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        purge_days: int = 63,
    ):
        self.train_ratio = train_ratio
        self.purge_days = purge_days

    def split(
        self,
        X: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data temporally with purging.

        Returns:
            (train_indices, test_indices)
        """
        dates = X.index.get_level_values("date") if isinstance(X.index, pd.MultiIndex) else X.index
        unique_dates = dates.unique().sort_values()

        n_train = int(len(unique_dates) * self.train_ratio)
        train_end_date = unique_dates[n_train - 1]
        test_start_date = unique_dates[min(n_train + self.purge_days, len(unique_dates) - 1)]

        train_mask = dates <= train_end_date
        test_mask = dates >= test_start_date

        return np.where(train_mask)[0], np.where(test_mask)[0]
