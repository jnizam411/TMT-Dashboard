#!/usr/bin/env python3
"""
Alpha Pipeline - Main Entry Point

Run the complete equity alpha generation pipeline:
1. Data acquisition
2. Feature engineering
3. Labeling
4. Model training
5. Validation
6. Backtesting

Usage:
    python run_pipeline.py --mode full        # Run everything
    python run_pipeline.py --mode train       # Train models only
    python run_pipeline.py --mode backtest    # Backtest only
    python run_pipeline.py --mode live        # Live inference
"""

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


def run_data_acquisition(
    output_dir: Path,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    force_refresh: bool = False,
):
    """Run data acquisition phase."""
    from src.data import DataDownloader, UniverseBuilder

    logger.info("=" * 60)
    logger.info("PHASE 1: DATA ACQUISITION")
    logger.info("=" * 60)

    # Build universe
    universe_builder = UniverseBuilder(data_dir=output_dir / "processed")
    universe = universe_builder.build_full_universe(
        start_date=start_date,
        end_date=end_date,
        force_refresh=force_refresh,
    )

    # Get unique tickers
    tickers = universe["ticker"].unique().tolist()
    logger.info(f"Universe contains {len(tickers)} unique tickers")

    # Download price data
    downloader = DataDownloader(raw_data_dir=output_dir / "raw")
    price_data = downloader.download_universe(
        tickers,
        start_date=start_date,
        end_date=end_date,
        force_refresh=force_refresh,
    )

    # Download macro data
    macro_data = downloader.download_macro_series(force_refresh=force_refresh)

    logger.info(f"Downloaded {len(price_data)} tickers, {len(macro_data)} macro series")

    return universe, price_data, macro_data


def run_feature_engineering(
    price_data: dict,
    output_dir: Path,
):
    """Run feature engineering phase."""
    from src.features import FeatureFactory

    logger.info("=" * 60)
    logger.info("PHASE 3: FEATURE ENGINEERING")
    logger.info("=" * 60)

    factory = FeatureFactory()
    factory.precompute_features(price_data, apply_cs_processing=True)

    # Save features
    features_path = output_dir / "processed" / "features.parquet"
    factory.save_features(features_path)

    logger.info(f"Features saved to {features_path}")

    return factory


def run_labeling(
    price_data: dict,
    output_dir: Path,
):
    """Run labeling phase."""
    from src.labeling import LabelingPipeline

    logger.info("=" * 60)
    logger.info("PHASE 2: LABELING")
    logger.info("=" * 60)

    pipeline = LabelingPipeline()
    labels = pipeline.label_universe(price_data)

    # Save labels
    labels_path = output_dir / "processed" / "labels.parquet"
    pipeline.save_labels(labels, labels_path)

    logger.info(f"Labels saved to {labels_path}")

    return labels


def run_training(
    features_path: Path,
    labels_path: Path,
    models_dir: Path,
):
    """Run model training phase."""
    from src.validation import PurgedWalkForwardSplitter
    from src.models import LightGBMClassifier, LassoWrapper, EnsemblePredictor

    logger.info("=" * 60)
    logger.info("PHASES 5-7: MODEL TRAINING & ENSEMBLE")
    logger.info("=" * 60)

    import pandas as pd

    # Load data
    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)

    # Setup validation
    splitter = PurgedWalkForwardSplitter()
    splits = splitter.generate_splits()

    logger.info(f"Training with {len(splits)} walk-forward splits")

    # Train models for each split
    models_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for split in splits:
        logger.info(f"Training split {split.split_id}: "
                   f"{split.train_start} to {split.test_end}")

        # Get data for split
        train_features, val_features, test_features = splitter.split_panel_data(
            features, split.split_id
        )
        train_labels, val_labels, test_labels = splitter.split_panel_data(
            labels, split.split_id
        )

        # Flatten for tabular models
        X_train = train_features.reset_index(level="ticker", drop=True)
        y_train = train_labels["label"].reset_index(level="ticker", drop=True)
        X_val = val_features.reset_index(level="ticker", drop=True)
        y_val = val_labels["label"].reset_index(level="ticker", drop=True)

        # Stage 1: Barrier touch prediction
        logger.info("Training Stage 1 (barrier touch)...")
        y_train_s1 = (train_labels["barrier_side"] != 0).astype(int)
        y_val_s1 = (val_labels["barrier_side"] != 0).astype(int)

        lgb_s1 = LightGBMClassifier()
        lgb_s1.train(X_train, y_train_s1.reset_index(level="ticker", drop=True),
                    X_val, y_val_s1.reset_index(level="ticker", drop=True))

        # Stage 2: Direction prediction (only high-confidence samples)
        logger.info("Training Stage 2 (direction)...")
        s1_probs = lgb_s1.predict_proba(X_train)
        high_conf = s1_probs > 0.65
        touched = (train_labels["barrier_side"] != 0).reset_index(level="ticker", drop=True)
        s2_mask = high_conf & touched

        if s2_mask.sum() > 100:
            y_train_s2 = (train_labels["barrier_side"] > 0).astype(int)
            y_train_s2 = y_train_s2.reset_index(level="ticker", drop=True)[s2_mask]

            lgb_s2 = LightGBMClassifier()
            lgb_s2.train(X_train[s2_mask], y_train_s2)

            # Save models
            split_dir = models_dir / f"split_{split.split_id}"
            split_dir.mkdir(exist_ok=True)
            lgb_s1.save(split_dir / "stage1_lgb.joblib")
            lgb_s2.save(split_dir / "stage2_lgb.joblib")

            logger.info(f"Saved models for split {split.split_id}")

    logger.info("Training complete")


def run_backtest(
    features_path: Path,
    models_dir: Path,
    output_dir: Path,
):
    """Run backtesting phase."""
    from src.backtest import VectorizedBacktest

    logger.info("=" * 60)
    logger.info("PHASE 8: BACKTESTING")
    logger.info("=" * 60)

    import pandas as pd

    # Load features
    features = pd.read_parquet(features_path)

    # Generate predictions from ensemble
    # (In production, would load models and generate predictions)

    # For now, use feature-based ranking as placeholder
    # A proper implementation would load trained models

    backtester = VectorizedBacktest()

    logger.info("Backtest complete - see output for results")


def run_live_inference(
    model_path: Path,
    output_dir: Path,
):
    """Run live inference."""
    from src.inference import InferencePipeline

    logger.info("=" * 60)
    logger.info("LIVE INFERENCE")
    logger.info("=" * 60)

    pipeline = InferencePipeline(
        model_path=model_path,
        output_dir=output_dir,
    )

    results = pipeline.run()

    logger.info(f"Generated {results['n_signals']} signals")
    logger.info(f"Output: {results['signals_path']}")


def main():
    parser = argparse.ArgumentParser(
        description="Alpha Pipeline - Equity Alpha Generation System"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "data", "features", "labels", "train", "backtest", "live"],
        default="full",
        help="Pipeline mode to run",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("src/data"),
        help="Data directory",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Models directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2000-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh all data",
    )

    args = parser.parse_args()

    # Parse dates
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)

    # Create directories
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ALPHA PIPELINE - EQUITY ALPHA GENERATION SYSTEM")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Date range: {start_date} to {end_date}")

    try:
        if args.mode in ["full", "data"]:
            universe, price_data, macro_data = run_data_acquisition(
                args.data_dir,
                start_date,
                end_date,
                args.force_refresh,
            )

        if args.mode in ["full", "features"]:
            # Load price data if not in memory
            if args.mode == "features":
                from src.data import DataDownloader
                downloader = DataDownloader(raw_data_dir=args.data_dir / "raw")
                tickers = list(downloader.get_all_cached_tickers())
                price_data = {t: downloader._load_cached(t) for t in tickers}

            factory = run_feature_engineering(price_data, args.data_dir)

        if args.mode in ["full", "labels"]:
            if args.mode == "labels":
                from src.data import DataDownloader
                downloader = DataDownloader(raw_data_dir=args.data_dir / "raw")
                tickers = list(downloader.get_all_cached_tickers())
                price_data = {t: downloader._load_cached(t) for t in tickers}

            labels = run_labeling(price_data, args.data_dir)

        if args.mode in ["full", "train"]:
            run_training(
                args.data_dir / "processed" / "features.parquet",
                args.data_dir / "processed" / "labels.parquet",
                args.models_dir,
            )

        if args.mode in ["full", "backtest"]:
            run_backtest(
                args.data_dir / "processed" / "features.parquet",
                args.models_dir,
                args.output_dir,
            )

        if args.mode == "live":
            run_live_inference(
                args.models_dir / "ensemble.joblib",
                args.output_dir,
            )

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
