"""
Alpha Pipeline - Enterprise-Grade Equity Alpha Generation System

A fully purged, walk-forward validated, hybrid-ensemble daily U.S. equity
alpha generation pipeline designed for Apple Silicon Macs.

Target: Live cost-adjusted Sharpe ratios > 2.0
"""

__version__ = "1.0.0"
__author__ = "Alpha Pipeline Team"

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CONFIG_DIR = PROJECT_ROOT / "src" / "config"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
