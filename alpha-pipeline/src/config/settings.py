"""
Pydantic-based configuration management for Alpha Pipeline.

All parameters are battle-tested defaults that should not be changed
without thorough backtesting.
"""

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class DataConfig(BaseModel):
    """Data acquisition and processing configuration."""

    # Date range
    start_date: date = Field(default=date(2000, 1, 1), description="Backtest start date")
    end_date: date = Field(default=date(2025, 12, 31), description="Backtest end date")

    # Universe configuration
    universe_update_frequency: str = Field(default="quarterly", description="How often to update universe")
    min_price: float = Field(default=5.0, description="Minimum stock price filter")
    min_adv: float = Field(default=1_000_000, description="Minimum average daily volume in USD")

    # Data sources
    primary_source: str = Field(default="yfinance", description="Primary data source")
    fallback_source: str = Field(default="alpha_vantage", description="Fallback data source")

    # Macro tickers
    macro_tickers: List[str] = Field(
        default=[
            "^GSPC", "^IXIC", "^RUT", "^VIX", "^TNX",
            "^MOVE", "DX-Y.NYB", "GLD", "USO", "TLT"
        ],
        description="Macro/intermarket series tickers"
    )


class LabelingConfig(BaseModel):
    """Triple barrier and meta-labeling configuration."""

    # Triple barrier parameters
    horizontal_barrier_days: int = Field(default=10, description="Time horizon in trading days")
    vertical_barrier_multiplier: float = Field(default=1.5, description="Multiplier on 20-day realized vol")
    vol_lookback: int = Field(default=20, description="Lookback for volatility calculation")
    min_return_threshold: float = Field(default=0.005, description="50 bps minimum for non-zero label")

    # Meta-labeling parameters
    stage1_threshold: float = Field(default=0.65, description="Probability threshold for stage 1")


class FeatureConfig(BaseModel):
    """Feature engineering configuration."""

    # Fractional differentiation
    frac_diff_d_values: List[float] = Field(
        default=[0.2, 0.3, 0.4, 0.5],
        description="Differentiation orders to compute"
    )
    frac_diff_threshold: float = Field(default=1e-5, description="Weight threshold for fracdiff")

    # Rolling windows
    return_windows: List[int] = Field(
        default=[5, 10, 21, 63, 126, 252],
        description="Windows for return-based features"
    )
    vol_windows: List[int] = Field(
        default=[63, 252],
        description="Windows for volatility features"
    )

    # Cross-sectional processing
    winsorize_lower: float = Field(default=0.05, description="Lower winsorization percentile")
    winsorize_upper: float = Field(default=0.95, description="Upper winsorization percentile")
    clip_sigma: float = Field(default=3.0, description="Clip features to ±N sigma")

    # VPIN parameters
    vpin_buckets: int = Field(default=20, description="Volume buckets per day for VPIN")


class ValidationConfig(BaseModel):
    """Purged walk-forward validation configuration."""

    # Split parameters
    train_start: date = Field(default=date(2005, 1, 1))
    train_end: date = Field(default=date(2015, 12, 31))
    purge_days: int = Field(default=63, description="Trading days to purge between train/val")
    val_days: int = Field(default=252, description="Validation period in trading days")
    test_days: int = Field(default=252, description="Test period in trading days")
    roll_days: int = Field(default=252, description="Days to roll forward each iteration")


class ModelConfig(BaseModel):
    """Model hyperparameters configuration."""

    # TFT parameters
    tft_sequence_length: int = Field(default=126)
    tft_hidden_size: int = Field(default=128)
    tft_quantiles: List[float] = Field(default=[0.1, 0.5, 0.9])
    tft_learning_rate: float = Field(default=0.001)
    tft_batch_size: int = Field(default=64)
    tft_max_epochs: int = Field(default=100)

    # N-BEATS parameters
    nbeats_lookback: int = Field(default=126)
    nbeats_horizon: int = Field(default=10)
    nbeats_stacks: int = Field(default=8)
    nbeats_hidden: int = Field(default=512)

    # LightGBM parameters
    lgb_n_estimators: int = Field(default=5000)
    lgb_learning_rate: float = Field(default=0.01)
    lgb_max_depth: int = Field(default=-1)
    lgb_num_leaves: int = Field(default=256)
    lgb_feature_fraction: float = Field(default=0.8)
    lgb_bagging_fraction: float = Field(default=0.8)
    lgb_lambda_l1: float = Field(default=1.0)
    lgb_early_stopping_rounds: int = Field(default=50)

    # Lasso parameters
    lasso_n_features: int = Field(default=40, description="Top N features for stability selection")
    lasso_alpha: float = Field(default=0.01)


class BacktestConfig(BaseModel):
    """Backtest engine configuration."""

    # Transaction costs
    fixed_cost_bps: float = Field(default=7.5, description="Fixed cost per trade in bps")
    impact_bps_per_mm: float = Field(default=0.1, description="Market impact per $1M notional")

    # Position limits
    max_position_adv_pct: float = Field(default=0.05, description="Max position as % of ADV")
    max_sector_weight: float = Field(default=0.25, description="Max sector weight")

    # Portfolio construction
    long_short_decile: int = Field(default=1, description="Decile for long/short (1=decile, 5=quintile)")
    rebalance_frequency: str = Field(default="daily")
    sector_neutral: bool = Field(default=True, description="Enforce sector neutrality")


class InferenceConfig(BaseModel):
    """Live inference configuration."""

    run_time: str = Field(default="16:30", description="Daily run time ET")
    output_dir: Path = Field(default=Path("signals"))
    min_ic_alert_threshold: float = Field(default=0.03, description="Alert if IC drops below this")
    consecutive_days_alert: int = Field(default=3, description="Days of low IC before alert")


class Settings(BaseSettings):
    """Main settings class aggregating all configurations."""

    # Sub-configurations
    data: DataConfig = Field(default_factory=DataConfig)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    # Global settings
    device: str = Field(default="mps", description="Compute device (mps, cuda, cpu)")
    random_seed: int = Field(default=42)
    n_jobs: int = Field(default=-1, description="Parallel jobs (-1 = all cores)")
    log_level: str = Field(default="INFO")

    # Paths
    project_root: Path = Field(default=Path(__file__).parent.parent.parent)

    model_config = {"env_prefix": "ALPHA_", "env_nested_delimiter": "__"}

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        """Load settings from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: Path) -> None:
        """Save settings to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    config_path = Path(__file__).parent / "defaults.yaml"
    if config_path.exists():
        return Settings.from_yaml(config_path)
    return Settings()
