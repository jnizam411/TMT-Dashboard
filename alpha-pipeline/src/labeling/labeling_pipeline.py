"""
End-to-end labeling pipeline.

Combines triple barrier labeling with meta-labeling in a single
reproducible pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from tqdm import tqdm

from ..config import get_settings
from .triple_barrier import BatchTripleBarrierLabeler, TripleBarrierLabeler
from .meta_labeling import MetaLabeler, MetaLabelingTrainer

logger = structlog.get_logger(__name__)


class LabelingPipeline:
    """
    Complete labeling pipeline for the alpha system.

    Handles:
    - Triple barrier label generation
    - Sample weight computation
    - Stage 1/2 label preparation
    - Purging for train/test splits
    """

    def __init__(
        self,
        horizontal_barrier_days: int = 10,
        vertical_barrier_multiplier: float = 1.5,
        vol_lookback: int = 20,
        min_return_threshold: float = 0.005,
        stage1_threshold: float = 0.65,
        use_batch_processing: bool = True,
        settings=None,
    ):
        self.settings = settings or get_settings()

        # Triple barrier parameters
        self.horizontal_barrier_days = horizontal_barrier_days
        self.vertical_barrier_multiplier = vertical_barrier_multiplier
        self.vol_lookback = vol_lookback
        self.min_return_threshold = min_return_threshold

        # Meta-labeling parameters
        self.stage1_threshold = stage1_threshold

        # Processing mode
        self.use_batch_processing = use_batch_processing

        # Initialize labelers
        if use_batch_processing:
            self.barrier_labeler = BatchTripleBarrierLabeler(
                horizontal_barrier_days=horizontal_barrier_days,
                vertical_barrier_multiplier=vertical_barrier_multiplier,
                vol_lookback=vol_lookback,
                min_return_threshold=min_return_threshold,
            )
        else:
            self.barrier_labeler = TripleBarrierLabeler(
                horizontal_barrier_days=horizontal_barrier_days,
                vertical_barrier_multiplier=vertical_barrier_multiplier,
                vol_lookback=vol_lookback,
                min_return_threshold=min_return_threshold,
            )

        self.meta_labeler = MetaLabeler(stage1_threshold=stage1_threshold)
        self.trainer = MetaLabelingTrainer(stage1_threshold=stage1_threshold)

    def label_single_ticker(
        self,
        df: pd.DataFrame,
        ticker: str = "",
        price_col: str = "close",
    ) -> pd.DataFrame:
        """
        Generate labels for a single ticker.

        Args:
            df: OHLCV DataFrame with DatetimeIndex
            ticker: Ticker symbol (for logging)
            price_col: Column to use for prices

        Returns:
            DataFrame with barrier events and labels
        """
        try:
            if self.use_batch_processing:
                events = self.barrier_labeler.get_barrier_events_batch(
                    df, price_col=price_col
                )
            else:
                events = self.barrier_labeler.get_barrier_events(
                    df, price_col=price_col
                )

            if len(events) > 0:
                events["ticker"] = ticker

            return events

        except Exception as e:
            logger.error(f"Failed to label {ticker}", error=str(e))
            return pd.DataFrame()

    def label_universe(
        self,
        price_data: Dict[str, pd.DataFrame],
        price_col: str = "close",
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Generate labels for entire universe.

        Args:
            price_data: Dict of ticker -> OHLCV DataFrame
            price_col: Column to use for prices
            show_progress: Show progress bar

        Returns:
            Combined DataFrame with all labels
        """
        all_events = []

        iterator = tqdm(price_data.items(), desc="Labeling") if show_progress else price_data.items()

        for ticker, df in iterator:
            events = self.label_single_ticker(df, ticker, price_col)
            if len(events) > 0:
                events = events.reset_index()
                all_events.append(events)

        if not all_events:
            return pd.DataFrame()

        combined = pd.concat(all_events, ignore_index=True)
        combined = combined.set_index(["t0", "ticker"])
        combined = combined.sort_index()

        # Log label distribution
        label_dist = combined["label"].value_counts(normalize=True)
        logger.info(f"Generated {len(combined)} labels",
                   tickers=len(price_data),
                   label_dist=label_dist.to_dict())

        return combined

    def compute_sample_weights(
        self,
        events: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
    ) -> pd.Series:
        """
        Compute sample weights accounting for overlapping labels.

        Args:
            events: Labeled events DataFrame
            price_data: Dict of ticker -> OHLCV DataFrame

        Returns:
            Series of sample weights
        """
        weights = []

        # Group by ticker
        if "ticker" in events.columns:
            grouped = events.reset_index().groupby("ticker")
        elif "ticker" in events.index.names:
            grouped = events.groupby(level="ticker")
        else:
            # Single ticker case
            ticker = list(price_data.keys())[0]
            close = price_data[ticker]["close"]
            return self.barrier_labeler.get_sample_weights(events, close)

        for ticker, group in grouped:
            if ticker not in price_data:
                continue

            close = price_data[ticker]["close"]

            # Reset index to just t0 for weight computation
            if isinstance(group.index, pd.MultiIndex):
                group = group.reset_index(level="ticker", drop=True)

            ticker_weights = self.barrier_labeler.get_sample_weights(group, close)
            ticker_weights = ticker_weights.to_frame()
            ticker_weights["ticker"] = ticker
            weights.append(ticker_weights)

        if not weights:
            return pd.Series()

        combined_weights = pd.concat(weights)
        combined_weights = combined_weights.set_index("ticker", append=True)
        combined_weights = combined_weights.iloc[:, 0]  # Get series

        return combined_weights

    def prepare_training_data(
        self,
        events: pd.DataFrame,
        features: pd.DataFrame,
        stage1_probs: Optional[pd.Series] = None,
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Prepare complete training datasets for both stages.

        Args:
            events: Labeled events DataFrame
            features: Feature matrix
            stage1_probs: Stage 1 predictions (for Stage 2 preparation)

        Returns:
            Dict with 'stage1' and optionally 'stage2' (X, y) tuples
        """
        result = {}

        # Stage 1 data
        X_stage1, y_stage1 = self.trainer.prepare_stage1_data(events, features)
        result["stage1"] = (X_stage1, y_stage1)

        # Stage 2 data (only if Stage 1 predictions provided)
        if stage1_probs is not None:
            X_stage2, y_stage2 = self.trainer.prepare_stage2_data(
                events, features, stage1_probs
            )
            result["stage2"] = (X_stage2, y_stage2)

        return result

    def save_labels(
        self,
        events: pd.DataFrame,
        path: Path,
    ):
        """Save labeled events to parquet."""
        events.to_parquet(path)
        logger.info(f"Saved {len(events)} labels to {path}")

    def load_labels(
        self,
        path: Path,
    ) -> pd.DataFrame:
        """Load labeled events from parquet."""
        events = pd.read_parquet(path)
        logger.info(f"Loaded {len(events)} labels from {path}")
        return events


class LabelingValidator:
    """
    Validation utilities for labeling quality.
    """

    @staticmethod
    def validate_labels(events: pd.DataFrame) -> Dict:
        """
        Validate label quality and return statistics.

        Returns:
            Dict with validation metrics
        """
        stats = {
            "n_samples": len(events),
            "label_distribution": events["label"].value_counts(normalize=True).to_dict(),
            "barrier_distribution": events["barrier_side"].value_counts(normalize=True).to_dict(),
            "avg_days_held": events["days_held"].mean(),
            "avg_return": events["return_at_touch"].mean(),
            "return_std": events["return_at_touch"].std(),
        }

        # Check for issues
        issues = []

        # Class imbalance
        label_dist = events["label"].value_counts(normalize=True)
        if any(label_dist < 0.1):
            issues.append("Severe class imbalance detected")

        # Too many time barrier hits
        time_barrier_pct = (events["barrier_side"] == 0).mean()
        if time_barrier_pct > 0.5:
            issues.append(f"High time barrier rate: {time_barrier_pct:.1%}")

        stats["issues"] = issues

        return stats

    @staticmethod
    def check_look_ahead_bias(
        events: pd.DataFrame,
        features: pd.DataFrame,
    ) -> bool:
        """
        Check for potential look-ahead bias in labels.

        Returns:
            True if no look-ahead bias detected
        """
        # Ensure all feature timestamps are before or equal to label timestamps
        if isinstance(events.index, pd.MultiIndex):
            event_times = events.index.get_level_values("t0")
        else:
            event_times = events.index

        feature_times = features.index

        # All feature times should be <= corresponding event times
        common_times = event_times.intersection(feature_times)

        if len(common_times) == 0:
            logger.warning("No common timestamps between events and features")
            return False

        logger.info("No look-ahead bias detected in label alignment")
        return True
