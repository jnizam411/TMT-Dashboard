"""
Live Data Loader for Inference.

Handles real-time data acquisition for live trading.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import structlog
import yfinance as yf
from tqdm import tqdm

from ..config import get_settings
from ..data.cleaners import DataCleaner
from ..features.pipeline import FeaturePipeline, CrossSectionalProcessor

logger = structlog.get_logger(__name__)


class LiveDataLoader:
    """
    Load live market data for inference.

    Designed to run at 4:30 PM ET after market close.
    """

    def __init__(
        self,
        lookback_days: int = 252,  # 1 year of history for features
        settings=None,
    ):
        """
        Initialize live data loader.

        Args:
            lookback_days: Days of historical data to fetch
        """
        self.settings = settings or get_settings()
        self.lookback_days = lookback_days

        self.cleaner = DataCleaner()
        self.feature_pipeline = FeaturePipeline()
        self.cs_processor = CrossSectionalProcessor()

    def load_universe(
        self,
        universe_file: Optional[Path] = None,
    ) -> List[str]:
        """
        Load current universe tickers.

        Args:
            universe_file: Path to universe parquet file

        Returns:
            List of ticker symbols
        """
        if universe_file is not None and universe_file.exists():
            df = pd.read_parquet(universe_file)
            today = date.today()

            # Get most recent universe
            df["date"] = pd.to_datetime(df["date"])
            latest = df[df["date"] == df["date"].max()]

            return latest[latest["in_universe"] == True]["ticker"].tolist()

        # Fallback to S&P 500 tickers
        logger.warning("Universe file not found, using S&P 500")
        return self._get_sp500_tickers()

    def _get_sp500_tickers(self) -> List[str]:
        """Get current S&P 500 tickers from Wikipedia."""
        try:
            import requests
            from bs4 import BeautifulSoup

            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response = requests.get(url, timeout=30)
            soup = BeautifulSoup(response.content, "lxml")

            table = soup.find("table", {"id": "constituents"})
            if table is None:
                table = soup.find("table", {"class": "wikitable"})

            tickers = []
            for row in table.find_all("tr")[1:]:
                cols = row.find_all("td")
                if cols:
                    ticker = cols[0].get_text(strip=True)
                    ticker = ticker.replace(".", "-")
                    tickers.append(ticker)

            return tickers

        except Exception as e:
            logger.error(f"Failed to fetch S&P 500 tickers: {e}")
            return []

    def fetch_live_data(
        self,
        tickers: List[str],
        end_date: Optional[date] = None,
        show_progress: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch live OHLCV data for universe.

        Args:
            tickers: List of ticker symbols
            end_date: End date (default: today)
            show_progress: Show progress bar

        Returns:
            Dict of ticker -> OHLCV DataFrame
        """
        end_date = end_date or date.today()
        start_date = end_date - timedelta(days=int(self.lookback_days * 1.5))  # Buffer

        data = {}
        failed = []

        iterator = tqdm(tickers, desc="Fetching live data") if show_progress else tickers

        for ticker in iterator:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(
                    start=start_date,
                    end=end_date + timedelta(days=1),
                    auto_adjust=False,
                )

                if df.empty:
                    failed.append(ticker)
                    continue

                # Standardize columns
                df.columns = df.columns.str.lower().str.replace(" ", "_")
                df = df[["open", "high", "low", "close", "adj_close", "volume"]]
                df.index = pd.to_datetime(df.index)
                df.index.name = "date"

                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                # Clean
                cleaned = self.cleaner.clean_ohlcv(df, ticker)
                if cleaned is not None:
                    data[ticker] = cleaned

            except Exception as e:
                logger.debug(f"Failed to fetch {ticker}: {e}")
                failed.append(ticker)

        if failed:
            logger.warning(f"Failed to fetch {len(failed)} tickers")

        logger.info(f"Fetched data for {len(data)} tickers")
        return data

    def compute_live_features(
        self,
        price_data: Dict[str, pd.DataFrame],
        target_date: Optional[date] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Compute features for live inference.

        Args:
            price_data: Dict of ticker -> OHLCV DataFrame
            target_date: Date to compute features for
            show_progress: Show progress bar

        Returns:
            DataFrame with features for target date
        """
        target_date = target_date or date.today()

        # Compute all features
        all_features = self.feature_pipeline.compute_features_universe(
            price_data,
            show_progress=show_progress,
        )

        # Apply cross-sectional processing
        processed = self.feature_pipeline.process_cross_sectionally(
            all_features,
            show_progress=show_progress,
        )

        # Get features for target date
        target_ts = pd.Timestamp(target_date)

        try:
            date_features = processed.xs(target_ts, level="date")
            logger.info(f"Computed features for {len(date_features)} stocks on {target_date}")
            return date_features
        except KeyError:
            # Try nearest date
            dates = processed.index.get_level_values("date").unique()
            nearest = dates[dates <= target_ts].max()
            date_features = processed.xs(nearest, level="date")
            logger.warning(f"Target date not found, using {nearest.date()}")
            return date_features

    def get_latest_prices(
        self,
        tickers: List[str],
    ) -> pd.Series:
        """Get latest closing prices."""
        prices = {}

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                if not hist.empty:
                    prices[ticker] = hist["Close"].iloc[-1]
            except Exception:
                continue

        return pd.Series(prices)

    def get_adv(
        self,
        price_data: Dict[str, pd.DataFrame],
        window: int = 21,
    ) -> pd.Series:
        """
        Compute average daily volume (in dollars).

        Args:
            price_data: Dict of ticker -> OHLCV DataFrame
            window: Lookback window

        Returns:
            Series of ADV by ticker
        """
        adv = {}

        for ticker, df in price_data.items():
            recent = df.tail(window)
            if len(recent) >= 5:
                dollar_volume = recent["close"] * recent["volume"]
                adv[ticker] = dollar_volume.mean()

        return pd.Series(adv)


class MarketDataValidator:
    """
    Validate market data quality before inference.
    """

    def __init__(
        self,
        min_price: float = 5.0,
        min_adv: float = 1_000_000,
        max_missing_pct: float = 0.1,
    ):
        self.min_price = min_price
        self.min_adv = min_adv
        self.max_missing_pct = max_missing_pct

    def validate(
        self,
        price_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """
        Validate and filter price data.

        Returns:
            Filtered price data passing all checks
        """
        validated = {}

        for ticker, df in price_data.items():
            # Check minimum price
            if df["close"].iloc[-1] < self.min_price:
                continue

            # Check ADV
            recent_adv = (df["close"] * df["volume"]).tail(21).mean()
            if recent_adv < self.min_adv:
                continue

            # Check missing data
            missing_pct = df.isna().mean().mean()
            if missing_pct > self.max_missing_pct:
                continue

            validated[ticker] = df

        logger.info(f"Validated {len(validated)}/{len(price_data)} tickers")
        return validated


def main():
    """Command-line entry point for live data loading."""
    import argparse

    parser = argparse.ArgumentParser(description="Load live market data")
    parser.add_argument("--output", type=Path, default=Path("live_features.parquet"))
    parser.add_argument("--universe", type=Path, default=None)

    args = parser.parse_args()

    loader = LiveDataLoader()

    # Load universe
    tickers = loader.load_universe(args.universe)
    logger.info(f"Loaded {len(tickers)} tickers")

    # Fetch data
    price_data = loader.fetch_live_data(tickers)

    # Validate
    validator = MarketDataValidator()
    price_data = validator.validate(price_data)

    # Compute features
    features = loader.compute_live_features(price_data)

    # Save
    features.to_parquet(args.output)
    logger.info(f"Saved features to {args.output}")


if __name__ == "__main__":
    main()
