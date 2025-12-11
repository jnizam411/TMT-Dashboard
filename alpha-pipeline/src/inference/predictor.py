"""
Live Predictor for Inference.

Generates trading signals using trained ensemble model.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

from ..config import get_settings
from ..models.ensemble import EnsemblePredictor

logger = structlog.get_logger(__name__)


class LivePredictor:
    """
    Generate live trading signals.

    Runs at 4:30 PM ET to generate next-day signals.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        settings=None,
    ):
        """
        Initialize live predictor.

        Args:
            model_path: Path to saved ensemble model
            output_dir: Directory for signal outputs
        """
        self.settings = settings or get_settings()
        self.model_path = model_path or Path("models/ensemble.joblib")
        self.output_dir = output_dir or Path(self.settings.inference.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ensemble: Optional[EnsemblePredictor] = None
        self.is_loaded = False

    def load_model(self):
        """Load trained ensemble model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.ensemble = EnsemblePredictor()
        self.ensemble.load(self.model_path)
        self.is_loaded = True

        logger.info(f"Loaded model from {self.model_path}")

    def predict(
        self,
        features: pd.DataFrame,
        volatility: Optional[pd.Series] = None,
        date_stamp: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Generate predictions for current features.

        Args:
            features: Feature DataFrame (tickers as index)
            volatility: Optional volatility series
            date_stamp: Date for output file

        Returns:
            DataFrame with signals
        """
        if not self.is_loaded:
            self.load_model()

        date_stamp = date_stamp or date.today()

        # Get scores
        scores = self.ensemble.predict_score(features, volatility)

        # Generate signals
        signals = self.ensemble.generate_signals(
            features,
            pd.Timestamp(date_stamp),
            volatility,
        )

        # Add metadata
        signals["generated_at"] = datetime.now().isoformat()
        signals["model_path"] = str(self.model_path)

        return signals

    def save_signals(
        self,
        signals: pd.DataFrame,
        date_stamp: Optional[date] = None,
    ) -> Path:
        """
        Save signals to CSV file.

        Args:
            signals: Signal DataFrame
            date_stamp: Date for filename

        Returns:
            Path to saved file
        """
        date_stamp = date_stamp or date.today()
        filename = f"signals_{date_stamp.strftime('%Y%m%d')}.csv"
        path = self.output_dir / filename

        signals.to_csv(path, index=False)
        logger.info(f"Saved signals to {path}")

        return path

    def run(
        self,
        features: pd.DataFrame,
        volatility: Optional[pd.Series] = None,
    ) -> Tuple[pd.DataFrame, Path]:
        """
        Run full prediction pipeline.

        Args:
            features: Feature DataFrame
            volatility: Optional volatility series

        Returns:
            (signals DataFrame, path to saved file)
        """
        signals = self.predict(features, volatility)
        path = self.save_signals(signals)

        # Log summary
        n_long = (signals["position"] == 1).sum()
        n_short = (signals["position"] == -1).sum()
        logger.info(f"Generated {n_long} long, {n_short} short signals")

        return signals, path


class SignalMonitor:
    """
    Monitor signal quality and alert on issues.
    """

    def __init__(
        self,
        min_ic_threshold: float = 0.03,
        consecutive_days_alert: int = 3,
        settings=None,
    ):
        """
        Initialize signal monitor.

        Args:
            min_ic_threshold: Alert if IC below this
            consecutive_days_alert: Days of low IC before alert
        """
        self.settings = settings or get_settings()
        self.min_ic_threshold = (
            min_ic_threshold or
            self.settings.inference.min_ic_alert_threshold
        )
        self.consecutive_days_alert = (
            consecutive_days_alert or
            self.settings.inference.consecutive_days_alert
        )

        self.ic_history: List[float] = []

    def update_ic(
        self,
        predictions: pd.Series,
        actual_returns: pd.Series,
    ) -> float:
        """
        Update IC history with new observation.

        Args:
            predictions: Model predictions
            actual_returns: Realized returns

        Returns:
            Today's IC
        """
        from scipy import stats

        # Align
        common = predictions.index.intersection(actual_returns.index)
        pred = predictions.loc[common]
        ret = actual_returns.loc[common]

        # Compute IC
        ic, _ = stats.spearmanr(pred.dropna(), ret.dropna())

        self.ic_history.append(ic)

        return ic

    def check_alerts(self) -> List[str]:
        """
        Check for alert conditions.

        Returns:
            List of alert messages
        """
        alerts = []

        if len(self.ic_history) >= self.consecutive_days_alert:
            recent_ic = self.ic_history[-self.consecutive_days_alert:]

            if all(ic < self.min_ic_threshold for ic in recent_ic):
                alerts.append(
                    f"ALERT: IC below {self.min_ic_threshold} for "
                    f"{self.consecutive_days_alert} consecutive days"
                )

        return alerts

    def send_alerts(
        self,
        alerts: List[str],
        slack_webhook: Optional[str] = None,
        email_recipients: Optional[List[str]] = None,
    ):
        """Send alerts via Slack and/or email."""
        if not alerts:
            return

        message = "\n".join(alerts)

        # Slack
        if slack_webhook:
            try:
                import requests
                requests.post(slack_webhook, json={"text": message})
                logger.info("Sent Slack alert")
            except Exception as e:
                logger.error(f"Failed to send Slack alert: {e}")

        # Email (would need SMTP config)
        if email_recipients:
            logger.warning("Email alerts not implemented")

        # Always log
        for alert in alerts:
            logger.warning(alert)


class InferencePipeline:
    """
    Complete inference pipeline.

    Runs daily at 4:30 PM ET:
    1. Fetch live data
    2. Compute features
    3. Generate predictions
    4. Save signals
    5. Monitor quality
    """

    def __init__(
        self,
        model_path: Path,
        universe_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        settings=None,
    ):
        self.settings = settings or get_settings()

        from .live_loader import LiveDataLoader

        self.data_loader = LiveDataLoader()
        self.predictor = LivePredictor(model_path, output_dir)
        self.monitor = SignalMonitor()

        self.universe_path = universe_path

    def run(self) -> Dict:
        """
        Run complete inference pipeline.

        Returns:
            Dict with results and metrics
        """
        logger.info("Starting inference pipeline...")
        start_time = datetime.now()

        # Load universe
        tickers = self.data_loader.load_universe(self.universe_path)
        logger.info(f"Universe: {len(tickers)} tickers")

        # Fetch data
        price_data = self.data_loader.fetch_live_data(tickers)

        # Compute features
        features = self.data_loader.compute_live_features(price_data)

        # Get volatility for scaling
        volatility = pd.Series({
            t: df["close"].pct_change().std()
            for t, df in price_data.items()
        })
        volatility = volatility.reindex(features.index)

        # Generate signals
        signals, signal_path = self.predictor.run(features, volatility)

        # Check alerts
        alerts = self.monitor.check_alerts()
        if alerts:
            self.monitor.send_alerts(alerts)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"Inference completed in {duration:.1f} seconds")

        return {
            "signals_path": signal_path,
            "n_signals": len(signals),
            "n_long": (signals["position"] == 1).sum(),
            "n_short": (signals["position"] == -1).sum(),
            "duration_seconds": duration,
            "alerts": alerts,
        }


def run_live():
    """Entry point for live inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Run live inference")
    parser.add_argument("--model", type=Path, required=True, help="Path to ensemble model")
    parser.add_argument("--universe", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("signals"))

    args = parser.parse_args()

    pipeline = InferencePipeline(
        model_path=args.model,
        universe_path=args.universe,
        output_dir=args.output,
    )

    results = pipeline.run()

    print(f"\nInference Results:")
    print(f"  Signals saved to: {results['signals_path']}")
    print(f"  Long positions: {results['n_long']}")
    print(f"  Short positions: {results['n_short']}")
    print(f"  Duration: {results['duration_seconds']:.1f}s")

    if results["alerts"]:
        print("\nAlerts:")
        for alert in results["alerts"]:
            print(f"  - {alert}")


if __name__ == "__main__":
    run_live()
