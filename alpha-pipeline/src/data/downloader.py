"""
Data acquisition module with idempotent, resumable downloads.

Primary source: yfinance
Fallback: Alpha Vantage free tier

All data cached in parquet format with integer timestamps.
"""

import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import structlog
import yfinance as yf
from tqdm import tqdm

from ..config import get_settings

logger = structlog.get_logger(__name__)


class DataDownloader:
    """
    Idempotent, resumable data downloader for equity and macro data.

    Features:
    - Automatic caching in parquet format
    - Resume capability for interrupted downloads
    - Rate limiting to avoid API bans
    - Proper error handling with retries
    """

    def __init__(
        self,
        raw_data_dir: Optional[Path] = None,
        settings=None,
    ):
        self.settings = settings or get_settings()
        self.raw_data_dir = raw_data_dir or Path(__file__).parent / "raw"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectory for macro data
        self.macro_dir = self.raw_data_dir / "macro"
        self.macro_dir.mkdir(exist_ok=True)

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.2  # 200ms between requests

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _get_cache_path(self, ticker: str, is_macro: bool = False) -> Path:
        """Get cache path for a ticker."""
        clean_ticker = ticker.replace("^", "").replace(".", "_").replace("-", "_")
        if is_macro:
            return self.macro_dir / f"{clean_ticker}.parquet"
        return self.raw_data_dir / f"{clean_ticker}.parquet"

    def _is_cached(self, ticker: str, is_macro: bool = False) -> bool:
        """Check if ticker data is already cached."""
        return self._get_cache_path(ticker, is_macro).exists()

    def _load_cached(self, ticker: str, is_macro: bool = False) -> Optional[pd.DataFrame]:
        """Load cached data if available."""
        cache_path = self._get_cache_path(ticker, is_macro)
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                logger.debug(f"Loaded cached data for {ticker}", rows=len(df))
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache for {ticker}", error=str(e))
        return None

    def _save_to_cache(self, ticker: str, df: pd.DataFrame, is_macro: bool = False):
        """Save data to cache."""
        cache_path = self._get_cache_path(ticker, is_macro)
        df.to_parquet(cache_path, index=True)
        logger.debug(f"Cached data for {ticker}", rows=len(df))

    def download_ticker(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        force_refresh: bool = False,
        is_macro: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Download OHLCV data for a single ticker.

        Args:
            ticker: Stock or index ticker symbol
            start_date: Start date (default: from settings)
            end_date: End date (default: from settings)
            force_refresh: If True, re-download even if cached
            is_macro: Whether this is a macro/index ticker

        Returns:
            DataFrame with columns [open, high, low, close, adj_close, volume]
            Index is DatetimeIndex with integer timestamp representation
        """
        start_date = start_date or self.settings.data.start_date
        end_date = end_date or self.settings.data.end_date

        # Check cache first
        if not force_refresh:
            cached = self._load_cached(ticker, is_macro)
            if cached is not None:
                # Check if cache covers requested date range
                cache_start = cached.index.min().date()
                cache_end = cached.index.max().date()
                if cache_start <= start_date and cache_end >= end_date:
                    return cached
                # Need to update cache
                logger.info(f"Cache incomplete for {ticker}, updating...")

        # Rate limit
        self._rate_limit()

        # Download from yfinance
        try:
            logger.info(f"Downloading {ticker} from yfinance",
                       start=str(start_date), end=str(end_date))

            stock = yf.Ticker(ticker)
            df = stock.history(
                start=start_date,
                end=end_date,
                auto_adjust=False,  # Keep both Close and Adj Close
            )

            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return None

            # Standardize column names
            df.columns = df.columns.str.lower().str.replace(" ", "_")

            # Keep only needed columns
            keep_cols = ["open", "high", "low", "close", "adj_close", "volume"]
            available_cols = [c for c in keep_cols if c in df.columns]
            df = df[available_cols]

            # Ensure index is DatetimeIndex
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"

            # Remove timezone info for consistency
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Add integer timestamp column for efficient merging
            df["timestamp"] = df.index.astype(np.int64) // 10**9

            # Save to cache
            self._save_to_cache(ticker, df, is_macro)

            logger.info(f"Downloaded {ticker}", rows=len(df))
            return df

        except Exception as e:
            logger.error(f"Failed to download {ticker}", error=str(e))
            return None

    def download_universe(
        self,
        tickers: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        force_refresh: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple tickers with progress tracking.

        Returns:
            Dictionary mapping ticker -> DataFrame
        """
        results = {}
        failed = []

        iterator = tqdm(tickers, desc="Downloading") if show_progress else tickers

        for ticker in iterator:
            df = self.download_ticker(
                ticker,
                start_date=start_date,
                end_date=end_date,
                force_refresh=force_refresh,
            )
            if df is not None:
                results[ticker] = df
            else:
                failed.append(ticker)

        if failed:
            logger.warning(f"Failed to download {len(failed)} tickers",
                          tickers=failed[:10])

        logger.info(f"Downloaded {len(results)}/{len(tickers)} tickers")
        return results

    def download_macro_series(
        self,
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download all macro/intermarket series.

        Returns:
            Dictionary mapping ticker -> DataFrame
        """
        macro_tickers = self.settings.data.macro_tickers
        results = {}

        for ticker in tqdm(macro_tickers, desc="Downloading macro"):
            df = self.download_ticker(
                ticker,
                force_refresh=force_refresh,
                is_macro=True,
            )
            if df is not None:
                results[ticker] = df

        logger.info(f"Downloaded {len(results)} macro series")
        return results

    def get_all_cached_tickers(self) -> Set[str]:
        """Get set of all tickers with cached data."""
        tickers = set()
        for path in self.raw_data_dir.glob("*.parquet"):
            if path.parent == self.raw_data_dir:  # Exclude macro subdir
                tickers.add(path.stem)
        return tickers

    def update_incremental(
        self,
        tickers: Optional[List[str]] = None,
    ) -> int:
        """
        Update existing cached data with latest available data.

        Returns:
            Number of tickers updated
        """
        if tickers is None:
            tickers = list(self.get_all_cached_tickers())

        updated = 0
        today = date.today()

        for ticker in tqdm(tickers, desc="Updating"):
            cached = self._load_cached(ticker)
            if cached is not None:
                last_date = cached.index.max().date()
                if last_date < today:
                    # Download only new data
                    df = self.download_ticker(
                        ticker,
                        start_date=last_date,
                        end_date=today,
                        force_refresh=True,
                    )
                    if df is not None and len(df) > 0:
                        # Merge with existing
                        combined = pd.concat([cached, df])
                        combined = combined[~combined.index.duplicated(keep="last")]
                        combined = combined.sort_index()
                        self._save_to_cache(ticker, combined)
                        updated += 1

        logger.info(f"Updated {updated} tickers")
        return updated


class AlphaVantageDownloader:
    """
    Fallback data source using Alpha Vantage free tier.

    Note: Free tier has strict rate limits (5 calls/minute, 500 calls/day)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or self._load_api_key()
        self._last_request_time = 0
        self._min_request_interval = 12.0  # 5 calls per minute

    def _load_api_key(self) -> Optional[str]:
        """Load API key from environment or file."""
        import os
        key = os.environ.get("ALPHA_VANTAGE_API_KEY")
        if key:
            return key

        # Check for key file
        key_file = Path.home() / ".alpha_vantage_key"
        if key_file.exists():
            return key_file.read_text().strip()

        logger.warning("Alpha Vantage API key not found")
        return None

    def _rate_limit(self):
        """Enforce strict rate limiting for free tier."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def download_ticker(
        self,
        ticker: str,
        outputsize: str = "full",
    ) -> Optional[pd.DataFrame]:
        """Download daily data from Alpha Vantage."""
        if not self.api_key:
            logger.error("No Alpha Vantage API key available")
            return None

        self._rate_limit()

        try:
            from alpha_vantage.timeseries import TimeSeries

            ts = TimeSeries(key=self.api_key, output_format="pandas")
            df, meta = ts.get_daily_adjusted(symbol=ticker, outputsize=outputsize)

            # Standardize columns
            df.columns = ["open", "high", "low", "close", "adj_close",
                         "volume", "dividend", "split_coef"]
            df = df[["open", "high", "low", "close", "adj_close", "volume"]]

            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            df = df.sort_index()

            logger.info(f"Downloaded {ticker} from Alpha Vantage", rows=len(df))
            return df

        except Exception as e:
            logger.error(f"Alpha Vantage failed for {ticker}", error=str(e))
            return None
