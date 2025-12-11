"""
Data cleaning and preprocessing utilities.

Handles missing data, corporate actions, and data quality issues.
"""

from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

from ..config import get_settings

logger = structlog.get_logger(__name__)


class DataCleaner:
    """
    Clean and preprocess OHLCV data.

    Handles:
    - Missing data interpolation
    - Outlier detection and removal
    - Corporate action adjustments
    - Data quality validation
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()

    def clean_ohlcv(
        self,
        df: pd.DataFrame,
        ticker: str = "",
        fill_method: str = "ffill",
        max_missing_pct: float = 0.1,
    ) -> Optional[pd.DataFrame]:
        """
        Clean a single ticker's OHLCV data.

        Args:
            df: Raw OHLCV DataFrame
            ticker: Ticker symbol for logging
            fill_method: Method for filling missing values
            max_missing_pct: Maximum allowed missing data percentage

        Returns:
            Cleaned DataFrame or None if data quality too poor
        """
        if df is None or df.empty:
            logger.warning(f"Empty data for {ticker}")
            return None

        df = df.copy()
        original_len = len(df)

        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Remove duplicate indices
        df = df[~df.index.duplicated(keep="last")]

        # Sort by date
        df = df.sort_index()

        # Check for required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns for {ticker}: {missing_cols}")
            return None

        # Handle negative values (data errors)
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df.loc[df[col] < 0, col] = np.nan

        # Fix OHLC relationships (high >= low, etc.)
        df = self._fix_ohlc_relationships(df)

        # Detect and handle outliers
        df = self._handle_price_outliers(df)

        # Check missing data percentage
        missing_pct = df[required_cols].isna().mean().mean()
        if missing_pct > max_missing_pct:
            logger.warning(f"Too much missing data for {ticker}: {missing_pct:.1%}")
            return None

        # Fill missing values
        if fill_method == "ffill":
            df = df.ffill()
        elif fill_method == "interpolate":
            df = df.interpolate(method="time")

        # Drop any remaining rows with missing prices
        df = df.dropna(subset=["close"])

        # Fill any remaining missing volume with 0
        df["volume"] = df["volume"].fillna(0)

        # Validate data
        if not self._validate_data(df, ticker):
            return None

        if len(df) < original_len * 0.5:
            logger.warning(f"Lost too much data for {ticker}: "
                          f"{original_len} -> {len(df)}")
            return None

        logger.debug(f"Cleaned {ticker}: {original_len} -> {len(df)} rows")
        return df

    def _fix_ohlc_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix inconsistent OHLC relationships."""
        # High should be >= max(open, close)
        df["high"] = df[["high", "open", "close"]].max(axis=1)

        # Low should be <= min(open, close)
        df["low"] = df[["low", "open", "close"]].min(axis=1)

        return df

    def _handle_price_outliers(
        self,
        df: pd.DataFrame,
        max_daily_return: float = 0.5,  # 50% max single day move
    ) -> pd.DataFrame:
        """Detect and handle price outliers."""
        returns = df["close"].pct_change()

        # Flag extreme returns
        extreme_mask = returns.abs() > max_daily_return

        if extreme_mask.any():
            extreme_dates = df.index[extreme_mask]
            logger.debug(f"Found {len(extreme_dates)} extreme price moves")

            # Replace extreme values with NaN (will be filled later)
            for col in ["open", "high", "low", "close"]:
                df.loc[extreme_mask, col] = np.nan

        return df

    def _validate_data(self, df: pd.DataFrame, ticker: str = "") -> bool:
        """Validate cleaned data quality."""
        if df.empty:
            return False

        # Check for any remaining NaN in prices
        price_cols = ["open", "high", "low", "close"]
        if df[price_cols].isna().any().any():
            logger.warning(f"Still have NaN prices for {ticker}")
            return False

        # Check for zero prices
        if (df[price_cols] == 0).any().any():
            logger.warning(f"Zero prices found for {ticker}")
            return False

        # Check for reasonable price range
        if df["close"].min() < 0.01 or df["close"].max() > 1_000_000:
            logger.warning(f"Suspicious price range for {ticker}")
            return False

        return True

    def clean_universe(
        self,
        price_data: Dict[str, pd.DataFrame],
        show_progress: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Clean all tickers in universe.

        Returns:
            Dictionary of cleaned DataFrames (excludes failed cleanings)
        """
        from tqdm import tqdm

        cleaned = {}
        failed = []

        iterator = tqdm(price_data.items(), desc="Cleaning") if show_progress else price_data.items()

        for ticker, df in iterator:
            cleaned_df = self.clean_ohlcv(df, ticker)
            if cleaned_df is not None:
                cleaned[ticker] = cleaned_df
            else:
                failed.append(ticker)

        logger.info(f"Cleaned {len(cleaned)}/{len(price_data)} tickers, "
                   f"{len(failed)} failed")

        return cleaned

    def align_to_trading_calendar(
        self,
        df: pd.DataFrame,
        trading_calendar: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Align data to a standard trading calendar."""
        df = df.reindex(trading_calendar)
        return df.ffill()

    def compute_returns(
        self,
        df: pd.DataFrame,
        price_col: str = "adj_close",
        return_types: List[str] = ["simple", "log"],
    ) -> pd.DataFrame:
        """
        Compute return series.

        Args:
            df: OHLCV DataFrame
            price_col: Column to use for returns (adj_close recommended)
            return_types: Types of returns to compute

        Returns:
            DataFrame with return columns added
        """
        df = df.copy()

        # Use adj_close if available, else close
        if price_col not in df.columns:
            price_col = "close"

        prices = df[price_col]

        if "simple" in return_types:
            df["return"] = prices.pct_change()

        if "log" in return_types:
            df["log_return"] = np.log(prices / prices.shift(1))

        return df

    def compute_forward_returns(
        self,
        df: pd.DataFrame,
        horizons: List[int] = [1, 5, 10, 21],
        price_col: str = "adj_close",
    ) -> pd.DataFrame:
        """
        Compute forward returns for various horizons.

        These are used for labels and model targets.
        """
        df = df.copy()

        if price_col not in df.columns:
            price_col = "close"

        prices = df[price_col]

        for h in horizons:
            df[f"fwd_return_{h}d"] = prices.shift(-h) / prices - 1

        return df


class PanelDataBuilder:
    """
    Build panel data structure for cross-sectional analysis.

    Combines multiple tickers into a single multi-indexed DataFrame
    suitable for cross-sectional feature processing and modeling.
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()

    def build_panel(
        self,
        price_data: Dict[str, pd.DataFrame],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Build panel DataFrame from individual ticker DataFrames.

        Returns:
            MultiIndex DataFrame with (date, ticker) index
        """
        start_date = start_date or self.settings.data.start_date
        end_date = end_date or self.settings.data.end_date

        panels = []

        for ticker, df in price_data.items():
            df = df.copy()

            # Filter date range
            mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            df = df.loc[mask]

            if df.empty:
                continue

            # Add ticker column
            df["ticker"] = ticker
            df = df.reset_index()

            panels.append(df)

        if not panels:
            return pd.DataFrame()

        panel = pd.concat(panels, ignore_index=True)
        panel = panel.set_index(["date", "ticker"])
        panel = panel.sort_index()

        logger.info(f"Built panel with {len(panel)} observations, "
                   f"{panel.index.get_level_values('ticker').nunique()} tickers")

        return panel

    def pivot_to_wide(
        self,
        panel: pd.DataFrame,
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Pivot panel to wide format (dates x tickers).

        Useful for cross-sectional calculations.
        """
        return panel[column].unstack(level="ticker")

    def unpivot_from_wide(
        self,
        wide_df: pd.DataFrame,
        column_name: str = "value",
    ) -> pd.DataFrame:
        """Convert wide format back to panel."""
        stacked = wide_df.stack()
        stacked.name = column_name
        return stacked.to_frame()
