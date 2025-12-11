"""
Meta-Labeling Implementation.

Meta-labeling is a two-stage approach:
1. Stage 1: Predict WHEN an opportunity exists (probability of barrier touch)
2. Stage 2: Predict WHICH direction (only on high-confidence opportunities)

This dramatically improves precision and reduces false positives.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import structlog

from ..config import get_settings

logger = structlog.get_logger(__name__)


class MetaLabeler:
    """
    Two-stage meta-labeling system.

    Stage 1 ("When"): Predicts probability of ANY barrier touch
    Stage 2 ("Which"): Predicts direction, ONLY when Stage 1 > threshold
    """

    def __init__(
        self,
        stage1_threshold: float = 0.65,
        settings=None,
    ):
        """
        Initialize meta-labeler.

        Args:
            stage1_threshold: Minimum Stage 1 probability to generate Stage 2 label
        """
        self.settings = settings or get_settings()
        self.stage1_threshold = (
            stage1_threshold or
            self.settings.labeling.stage1_threshold
        )

    def create_stage1_labels(
        self,
        events: pd.DataFrame,
    ) -> pd.Series:
        """
        Create binary labels for Stage 1 (barrier touch prediction).

        Returns:
            Series with 1 if barrier touched (upper or lower), 0 if time barrier
        """
        # Label is 1 if upper or lower barrier was touched
        stage1_labels = (events["barrier_side"] != 0).astype(int)
        stage1_labels.name = "stage1_label"

        touch_rate = stage1_labels.mean()
        logger.info(f"Stage 1 labels: {touch_rate:.1%} barrier touches")

        return stage1_labels

    def create_stage2_labels(
        self,
        events: pd.DataFrame,
        stage1_probs: pd.Series,
    ) -> pd.DataFrame:
        """
        Create directional labels for Stage 2.

        Only generates labels where:
        1. Stage 1 probability > threshold
        2. A barrier was actually touched (not time barrier)

        Args:
            events: Barrier events DataFrame
            stage1_probs: Predicted probabilities from Stage 1 model

        Returns:
            DataFrame with Stage 2 labels and metadata
        """
        # Align probabilities to events index
        aligned_probs = stage1_probs.reindex(events.index)

        # Filter conditions
        high_confidence = aligned_probs > self.stage1_threshold
        barrier_touched = events["barrier_side"] != 0

        # Combined filter
        valid_mask = high_confidence & barrier_touched

        # Create Stage 2 labels (direction)
        stage2_df = events.loc[valid_mask].copy()
        stage2_df["stage2_label"] = (events.loc[valid_mask, "barrier_side"] > 0).astype(int)
        stage2_df["stage1_prob"] = aligned_probs.loc[valid_mask]

        logger.info(f"Stage 2 labels: {len(stage2_df)}/{len(events)} events "
                   f"({len(stage2_df)/len(events):.1%})")

        return stage2_df

    def get_meta_labels(
        self,
        events: pd.DataFrame,
        predictions_stage1: pd.Series,
    ) -> pd.Series:
        """
        Get final meta-labels for trading.

        Returns +1/-1 only on days where:
        - Stage 1 probability > threshold
        - Barrier was actually touched

        Args:
            events: Barrier events DataFrame
            predictions_stage1: Stage 1 model probabilities

        Returns:
            Series with +1 (long), -1 (short), or NaN (no trade)
        """
        meta_labels = pd.Series(index=events.index, dtype=float)
        meta_labels[:] = np.nan

        # Get Stage 2 eligible events
        stage2_df = self.create_stage2_labels(events, predictions_stage1)

        # Assign directional labels
        for t0, row in stage2_df.iterrows():
            if row["barrier_side"] > 0:
                meta_labels.loc[t0] = 1  # Upper barrier hit -> long
            else:
                meta_labels.loc[t0] = -1  # Lower barrier hit -> short

        n_trades = meta_labels.notna().sum()
        logger.info(f"Generated {n_trades} meta-labels")

        return meta_labels


def get_meta_labels(
    events: pd.DataFrame,
    predictions_stage1: pd.Series,
    stage1_threshold: float = 0.65,
) -> pd.Series:
    """
    Convenience function for meta-labeling.

    Args:
        events: Barrier events from get_barrier_events()
        predictions_stage1: Stage 1 probability predictions
        stage1_threshold: Minimum probability threshold

    Returns:
        Series with +1 (long), -1 (short), or NaN (no trade)
    """
    labeler = MetaLabeler(stage1_threshold=stage1_threshold)
    return labeler.get_meta_labels(events, predictions_stage1)


class MetaLabelingTrainer:
    """
    Training utilities for the two-stage meta-labeling system.
    """

    def __init__(
        self,
        stage1_threshold: float = 0.65,
        settings=None,
    ):
        self.settings = settings or get_settings()
        self.stage1_threshold = stage1_threshold

    def prepare_stage1_data(
        self,
        events: pd.DataFrame,
        features: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data for Stage 1 model.

        Args:
            events: Barrier events with labels
            features: Feature matrix (aligned to events index)

        Returns:
            (X, y) for Stage 1 training
        """
        # Align features to events
        common_idx = events.index.intersection(features.index)

        X = features.loc[common_idx]
        y = (events.loc[common_idx, "barrier_side"] != 0).astype(int)

        logger.info(f"Stage 1 training data: {len(X)} samples, "
                   f"{y.mean():.1%} positive rate")

        return X, y

    def prepare_stage2_data(
        self,
        events: pd.DataFrame,
        features: pd.DataFrame,
        stage1_probs: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data for Stage 2 model.

        Only includes samples where Stage 1 probability > threshold.

        Args:
            events: Barrier events with labels
            features: Feature matrix
            stage1_probs: Stage 1 predicted probabilities

        Returns:
            (X, y) for Stage 2 training
        """
        meta_labeler = MetaLabeler(stage1_threshold=self.stage1_threshold)
        stage2_df = meta_labeler.create_stage2_labels(events, stage1_probs)

        # Align features to Stage 2 events
        common_idx = stage2_df.index.intersection(features.index)

        X = features.loc[common_idx]
        y = stage2_df.loc[common_idx, "stage2_label"]

        logger.info(f"Stage 2 training data: {len(X)} samples, "
                   f"{y.mean():.1%} positive rate")

        return X, y

    def compute_bet_size(
        self,
        stage1_prob: float,
        stage2_prob: float,
        max_size: float = 1.0,
    ) -> float:
        """
        Compute position size based on model confidence.

        Uses product of probabilities scaled to max size.

        Args:
            stage1_prob: Stage 1 probability (barrier touch)
            stage2_prob: Stage 2 probability (direction)
            max_size: Maximum position size

        Returns:
            Position size between 0 and max_size
        """
        if stage1_prob < self.stage1_threshold:
            return 0.0

        # Confidence-weighted size
        confidence = stage1_prob * abs(stage2_prob - 0.5) * 2
        size = min(confidence, max_size)

        return size


class PurgedMetaLabeler:
    """
    Meta-labeler with built-in purging for train/test leakage prevention.

    Ensures that when creating Stage 2 labels, we don't use information
    from overlapping barrier events.
    """

    def __init__(
        self,
        stage1_threshold: float = 0.65,
        purge_days: int = 63,
        settings=None,
    ):
        self.settings = settings or get_settings()
        self.stage1_threshold = stage1_threshold
        self.purge_days = purge_days

    def get_purged_train_test_split(
        self,
        events: pd.DataFrame,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split events into train/test with proper purging.

        Removes train events that overlap with test period.

        Args:
            events: All barrier events
            train_end: End of training period
            test_start: Start of test period

        Returns:
            (train_events, test_events) with purging applied
        """
        # Test events
        test_mask = events.index >= test_start
        test_events = events.loc[test_mask]

        # Train events - remove those overlapping with test
        train_mask = events.index <= train_end

        # Check for overlap with test period
        overlap_mask = events["t1"] >= test_start

        # Final train mask: before train_end AND no overlap with test
        final_train_mask = train_mask & ~overlap_mask

        train_events = events.loc[final_train_mask]

        n_purged = train_mask.sum() - final_train_mask.sum()
        logger.info(f"Purged {n_purged} overlapping events from training set")

        return train_events, test_events
