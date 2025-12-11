"""
Feature Pipeline and Factory.

Combines all feature families into a unified pipeline with
proper cross-sectional processing.

Cross-sectional processing is critical:
1. Winsorize at 5%/95%
2. Demean cross-sectionally
3. Divide by cross-sectional std
4. Clip to ±3 sigma
"""

from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import structlog
import torch
from tqdm import tqdm

from ..config import get_settings
from .fracdiff import FractionalDifferentiator
from .technical import TechnicalFeatures, MomentumFeatures
from .microstructure import MicrostructureFeatures
from .csi_vpin import CSICalculator, VPINCalculator, InformedTradingFeatures

logger = structlog.get_logger(__name__)


class CrossSectionalProcessor:
    """
    Cross-sectional feature processing.

    Applies standardization across all stocks on each day to ensure
    features are comparable and prevent outliers from dominating.
    """

    def __init__(
        self,
        winsorize_lower: float = 0.05,
        winsorize_upper: float = 0.95,
        clip_sigma: float = 3.0,
        settings=None,
    ):
        """
        Initialize cross-sectional processor.

        Args:
            winsorize_lower: Lower percentile for winsorization
            winsorize_upper: Upper percentile for winsorization
            clip_sigma: Clip to ±N standard deviations
        """
        self.settings = settings or get_settings()
        self.winsorize_lower = winsorize_lower or self.settings.features.winsorize_lower
        self.winsorize_upper = winsorize_upper or self.settings.features.winsorize_upper
        self.clip_sigma = clip_sigma or self.settings.features.clip_sigma

    def process_single_date(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply cross-sectional processing for a single date.

        Args:
            features: DataFrame with tickers as index, features as columns

        Returns:
            Processed DataFrame
        """
        result = features.copy()

        for col in result.columns:
            values = result[col]

            # Skip if all NaN
            if values.isna().all():
                continue

            # 1. Winsorize
            lower = values.quantile(self.winsorize_lower)
            upper = values.quantile(self.winsorize_upper)
            values = values.clip(lower=lower, upper=upper)

            # 2. Demean
            mean = values.mean()
            values = values - mean

            # 3. Divide by std
            std = values.std()
            if std > 0:
                values = values / std
            else:
                values = values * 0  # All same value

            # 4. Clip to ±N sigma
            values = values.clip(lower=-self.clip_sigma, upper=self.clip_sigma)

            result[col] = values

        return result

    def process_panel(
        self,
        features: pd.DataFrame,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Apply cross-sectional processing to panel data.

        Args:
            features: DataFrame with MultiIndex (date, ticker)

        Returns:
            Processed DataFrame
        """
        if not isinstance(features.index, pd.MultiIndex):
            raise ValueError("Expected MultiIndex (date, ticker)")

        dates = features.index.get_level_values("date").unique()
        processed_dfs = []

        iterator = tqdm(dates, desc="Cross-sectional processing") if show_progress else dates

        for dt in iterator:
            date_features = features.xs(dt, level="date")
            processed = self.process_single_date(date_features)
            processed["date"] = dt
            processed = processed.reset_index()
            processed_dfs.append(processed)

        result = pd.concat(processed_dfs, ignore_index=True)
        result = result.set_index(["date", "ticker"])
        result = result.sort_index()

        return result

    def process_wide_format(
        self,
        features_dict: Dict[str, pd.DataFrame],
        show_progress: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Process features in wide format (dates x tickers).

        Args:
            features_dict: Dict of feature_name -> DataFrame (dates as index, tickers as columns)

        Returns:
            Dict of processed DataFrames
        """
        result = {}

        for feature_name, df in tqdm(features_dict.items(),
                                     desc="Processing features",
                                     disable=not show_progress):
            processed_df = df.copy()

            for idx in processed_df.index:
                row = processed_df.loc[idx]

                # 1. Winsorize
                lower = row.quantile(self.winsorize_lower)
                upper = row.quantile(self.winsorize_upper)
                row = row.clip(lower=lower, upper=upper)

                # 2. Demean
                row = row - row.mean()

                # 3. Standardize
                std = row.std()
                if std > 0:
                    row = row / std

                # 4. Clip
                row = row.clip(lower=-self.clip_sigma, upper=self.clip_sigma)

                processed_df.loc[idx] = row

            result[feature_name] = processed_df

        return result


class FeaturePipeline:
    """
    Complete feature engineering pipeline.

    Combines all feature families and applies cross-sectional processing.
    """

    def __init__(
        self,
        settings=None,
    ):
        self.settings = settings or get_settings()

        # Feature calculators
        self.fracdiff = FractionalDifferentiator(settings=settings)
        self.technical = TechnicalFeatures(settings=settings)
        self.momentum = MomentumFeatures()
        self.microstructure = MicrostructureFeatures(settings=settings)
        self.csi = CSICalculator(settings=settings)
        self.vpin = VPINCalculator(settings=settings)
        self.informed_trading = InformedTradingFeatures(settings=settings)

        # Cross-sectional processor
        self.cs_processor = CrossSectionalProcessor(settings=settings)

    def compute_features_single_ticker(
        self,
        df: pd.DataFrame,
        ticker: str = "",
        include_fracdiff: bool = True,
    ) -> pd.DataFrame:
        """
        Compute all features for a single ticker.

        Args:
            df: OHLCV DataFrame
            ticker: Ticker symbol
            include_fracdiff: Include fractional differentiation features

        Returns:
            DataFrame with all features
        """
        feature_dfs = []

        # Ensure returns are computed
        if "return" not in df.columns:
            df = self.technical.compute_returns(df)

        # Technical features
        try:
            tech = self.technical.compute_all(df, include_returns=False)
            feature_dfs.append(tech)
        except Exception as e:
            logger.warning(f"Technical features failed for {ticker}: {e}")

        # Momentum features
        try:
            mom = self.momentum.compute_momentum(df)
            trend = self.momentum.compute_trend_strength(df)
            feature_dfs.append(mom)
            feature_dfs.append(trend)
        except Exception as e:
            logger.warning(f"Momentum features failed for {ticker}: {e}")

        # Microstructure features
        try:
            micro = self.microstructure.compute_all(df)
            feature_dfs.append(micro)
        except Exception as e:
            logger.warning(f"Microstructure features failed for {ticker}: {e}")

        # CSI and VPIN features
        try:
            informed = self.informed_trading.compute_all(df)
            feature_dfs.append(informed)
        except Exception as e:
            logger.warning(f"CSI/VPIN features failed for {ticker}: {e}")

        # Fractional differentiation
        if include_fracdiff:
            try:
                log_prices = np.log(df["close"])
                fracdiff_features = self.fracdiff.transform(log_prices, "log_price")
                feature_dfs.append(fracdiff_features)
            except Exception as e:
                logger.warning(f"Fracdiff features failed for {ticker}: {e}")

        # Combine all features
        if not feature_dfs:
            return pd.DataFrame(index=df.index)

        result = pd.concat(feature_dfs, axis=1)

        # Remove duplicate columns
        result = result.loc[:, ~result.columns.duplicated()]

        # Add metadata
        result["ticker"] = ticker

        return result

    def compute_features_universe(
        self,
        price_data: Dict[str, pd.DataFrame],
        show_progress: bool = True,
        include_fracdiff: bool = True,
    ) -> pd.DataFrame:
        """
        Compute features for entire universe.

        Args:
            price_data: Dict of ticker -> OHLCV DataFrame
            show_progress: Show progress bar
            include_fracdiff: Include fractional differentiation

        Returns:
            Panel DataFrame with MultiIndex (date, ticker)
        """
        all_features = []

        iterator = tqdm(price_data.items(), desc="Computing features") if show_progress else price_data.items()

        for ticker, df in iterator:
            features = self.compute_features_single_ticker(
                df, ticker, include_fracdiff
            )
            if len(features) > 0:
                features = features.reset_index()
                features = features.rename(columns={"index": "date"})
                all_features.append(features)

        if not all_features:
            return pd.DataFrame()

        combined = pd.concat(all_features, ignore_index=True)
        combined = combined.set_index(["date", "ticker"])
        combined = combined.sort_index()

        logger.info(f"Computed features for {len(price_data)} tickers, "
                   f"{len(combined.columns)} features")

        return combined

    def process_cross_sectionally(
        self,
        features: pd.DataFrame,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Apply cross-sectional processing to features.
        """
        return self.cs_processor.process_panel(features, show_progress)


class FeatureFactory:
    """
    Factory for generating feature tensors ready for model input.

    Produces (N × F) tensors on the target device (MPS/CUDA/CPU).
    """

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        device: str = "mps",
        settings=None,
    ):
        """
        Initialize feature factory.

        Args:
            feature_columns: List of features to include (None = all)
            device: Target device for tensors
        """
        self.settings = settings or get_settings()
        self.feature_columns = feature_columns
        self.device = device or self.settings.device

        self.pipeline = FeaturePipeline(settings=settings)
        self._feature_cache = {}

        # Determine device
        self._torch_device = self._get_torch_device()

    def _get_torch_device(self) -> torch.device:
        """Get appropriate torch device."""
        if self.device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def precompute_features(
        self,
        price_data: Dict[str, pd.DataFrame],
        apply_cs_processing: bool = True,
        show_progress: bool = True,
    ):
        """
        Precompute and cache all features.

        Args:
            price_data: Dict of ticker -> OHLCV DataFrame
            apply_cs_processing: Apply cross-sectional processing
            show_progress: Show progress bar
        """
        logger.info("Precomputing features...")

        # Compute raw features
        features = self.pipeline.compute_features_universe(
            price_data, show_progress
        )

        # Apply cross-sectional processing
        if apply_cs_processing:
            features = self.pipeline.process_cross_sectionally(
                features, show_progress
            )

        # Set feature columns if not specified
        if self.feature_columns is None:
            self.feature_columns = [
                c for c in features.columns
                if c not in ["ticker", "date"]
            ]

        self._feature_cache["features"] = features
        logger.info(f"Cached {len(features)} feature rows, {len(self.feature_columns)} features")

    def get_features_for_date(
        self,
        target_date: Union[date, pd.Timestamp],
        tickers: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Get feature tensor for a specific date.

        Args:
            target_date: Target date
            tickers: List of tickers to include (None = all available)

        Returns:
            (tensor of shape (N, F), list of ticker names)
        """
        if "features" not in self._feature_cache:
            raise ValueError("Features not precomputed. Call precompute_features first.")

        features = self._feature_cache["features"]

        # Convert date
        if isinstance(target_date, date):
            target_date = pd.Timestamp(target_date)

        # Get features for date
        try:
            date_features = features.xs(target_date, level="date")
        except KeyError:
            logger.warning(f"No features for date {target_date}")
            return torch.empty(0, len(self.feature_columns)), []

        # Filter tickers
        if tickers is not None:
            available_tickers = date_features.index.intersection(tickers)
            date_features = date_features.loc[available_tickers]

        if len(date_features) == 0:
            return torch.empty(0, len(self.feature_columns)), []

        # Get feature columns
        feature_values = date_features[self.feature_columns].values

        # Handle NaN
        feature_values = np.nan_to_num(feature_values, nan=0.0)

        # Convert to tensor
        tensor = torch.tensor(
            feature_values,
            dtype=torch.float32,
            device=self._torch_device
        )

        ticker_list = date_features.index.tolist()

        return tensor, ticker_list

    def get_feature_matrix(
        self,
        start_date: date,
        end_date: date,
        tickers: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
        """
        Get 3D feature matrix for date range.

        Returns:
            (array of shape (T, N, F), dates, tickers)
        """
        if "features" not in self._feature_cache:
            raise ValueError("Features not precomputed. Call precompute_features first.")

        features = self._feature_cache["features"]

        # Filter date range
        mask = (
            (features.index.get_level_values("date") >= pd.Timestamp(start_date)) &
            (features.index.get_level_values("date") <= pd.Timestamp(end_date))
        )
        filtered = features.loc[mask]

        if len(filtered) == 0:
            return np.array([]), pd.DatetimeIndex([]), []

        # Get unique dates and tickers
        dates = filtered.index.get_level_values("date").unique().sort_values()

        if tickers is None:
            tickers = filtered.index.get_level_values("ticker").unique().tolist()

        n_dates = len(dates)
        n_tickers = len(tickers)
        n_features = len(self.feature_columns)

        # Initialize 3D array
        matrix = np.full((n_dates, n_tickers, n_features), np.nan)

        # Fill matrix
        ticker_idx = {t: i for i, t in enumerate(tickers)}

        for t, dt in enumerate(dates):
            try:
                date_features = filtered.xs(dt, level="date")
                for ticker in date_features.index:
                    if ticker in ticker_idx:
                        matrix[t, ticker_idx[ticker], :] = date_features.loc[ticker, self.feature_columns].values
            except KeyError:
                continue

        return matrix, dates, tickers

    def get_sequential_features(
        self,
        target_date: date,
        sequence_length: int = 126,
        tickers: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Get sequential feature tensor for time series models.

        Returns:
            tensor of shape (N, T, F) for each ticker
        """
        end_date = target_date
        start_date = target_date - pd.Timedelta(days=sequence_length * 2)  # Buffer for weekends

        matrix, dates, available_tickers = self.get_feature_matrix(
            start_date, end_date, tickers
        )

        # Get last sequence_length dates
        if len(dates) < sequence_length:
            logger.warning(f"Not enough dates for sequence: {len(dates)}")
            return torch.empty(0, sequence_length, len(self.feature_columns)), []

        matrix = matrix[-sequence_length:]

        # Transpose to (N, T, F)
        matrix = matrix.transpose(1, 0, 2)

        # Handle NaN
        matrix = np.nan_to_num(matrix, nan=0.0)

        tensor = torch.tensor(
            matrix,
            dtype=torch.float32,
            device=self._torch_device
        )

        return tensor, available_tickers

    def save_features(self, path: Path):
        """Save precomputed features to disk."""
        if "features" not in self._feature_cache:
            raise ValueError("No features to save")

        self._feature_cache["features"].to_parquet(path)
        logger.info(f"Saved features to {path}")

    def load_features(self, path: Path):
        """Load precomputed features from disk."""
        self._feature_cache["features"] = pd.read_parquet(path)

        if self.feature_columns is None:
            self.feature_columns = [
                c for c in self._feature_cache["features"].columns
                if c not in ["ticker", "date"]
            ]

        logger.info(f"Loaded features from {path}")
