#!/usr/bin/env python3
"""
Live Inference Script

Runs at 4:30 PM ET to generate next-day trading signals.

Usage:
    python run_live.py --model models/ensemble.joblib --output signals/
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(
        description="Alpha Pipeline - Live Inference"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/ensemble.joblib"),
        help="Path to ensemble model",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=None,
        help="Path to universe file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("signals"),
        help="Output directory for signals",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ALPHA PIPELINE - LIVE INFERENCE")
    logger.info("=" * 60)
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output}")

    try:
        from src.inference import InferencePipeline

        pipeline = InferencePipeline(
            model_path=args.model,
            universe_path=args.universe,
            output_dir=args.output,
        )

        results = pipeline.run()

        logger.info("=" * 60)
        logger.info("INFERENCE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Signals: {results['signals_path']}")
        logger.info(f"Long: {results['n_long']}")
        logger.info(f"Short: {results['n_short']}")
        logger.info(f"Duration: {results['duration_seconds']:.1f}s")

        if results["alerts"]:
            logger.warning("ALERTS:")
            for alert in results["alerts"]:
                logger.warning(f"  {alert}")

    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        logger.info("Train models first with: python run_pipeline.py --mode train")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
