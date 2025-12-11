"""
Performance Metrics for Alpha Evaluation.

Key metrics:
- Information Coefficient (IC) and ICIR
- Sharpe Ratio (gross and net)
- Maximum Drawdown
- Turnover

All metrics must be computed on out-of-sample data only.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from scipy import stats

from ..config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""

    # IC metrics
    mean_ic: float
    icir: float
    ic_pvalue: float
    ic_hit_rate: float  # % of days with positive IC

    # Return metrics
    gross_sharpe: float
    net_sharpe: float
    total_return: float
    annualized_return: float

    # Risk metrics
    max_drawdown: float
    volatility: float
    calmar_ratio: float

    # Turnover
    mean_turnover: float
    turnover_cost: float

    # Additional
    n_days: int
    win_rate: float


def compute_ic(
    predictions: pd.Series,
    forward_returns: pd.Series,
    method: str = "spearman",
) -> float:
    """
    Compute Information Coefficient.

    IC = correlation between predictions and forward returns

    Args:
        predictions: Model predictions (scores/rankings)
        forward_returns: Actual forward returns
        method: "spearman" (rank) or "pearson"

    Returns:
        IC value
    """
    # Align data
    common_idx = predictions.index.intersection(forward_returns.index)
    pred = predictions.loc[common_idx]
    ret = forward_returns.loc[common_idx]

    # Remove NaN
    mask = ~(pred.isna() | ret.isna())
    pred = pred[mask]
    ret = ret[mask]

    if len(pred) < 10:
        return np.nan

    if method == "spearman":
        ic, _ = stats.spearmanr(pred, ret)
    else:
        ic, _ = stats.pearsonr(pred, ret)

    return ic


def compute_daily_ic(
    predictions: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """
    Compute IC for each day in cross-sectional data.

    Args:
        predictions: DataFrame with (date, ticker) index or wide format
        forward_returns: DataFrame with same format

    Returns:
        Series of daily IC values
    """
    # Handle different input formats
    if isinstance(predictions.index, pd.MultiIndex):
        dates = predictions.index.get_level_values("date").unique()
        ic_series = []

        for dt in dates:
            try:
                pred_day = predictions.xs(dt, level="date")
                ret_day = forward_returns.xs(dt, level="date")

                common = pred_day.index.intersection(ret_day.index)
                if len(common) < 10:
                    ic_series.append(np.nan)
                    continue

                pred = pred_day.loc[common].values.flatten()
                ret = ret_day.loc[common].values.flatten()

                mask = ~(np.isnan(pred) | np.isnan(ret))
                if mask.sum() < 10:
                    ic_series.append(np.nan)
                    continue

                if method == "spearman":
                    ic, _ = stats.spearmanr(pred[mask], ret[mask])
                else:
                    ic, _ = stats.pearsonr(pred[mask], ret[mask])

                ic_series.append(ic)

            except Exception:
                ic_series.append(np.nan)

        return pd.Series(ic_series, index=dates)

    else:
        # Wide format (dates as index, tickers as columns)
        ic_series = []

        for dt in predictions.index:
            pred = predictions.loc[dt].dropna()
            ret = forward_returns.loc[dt].dropna()

            common = pred.index.intersection(ret.index)
            if len(common) < 10:
                ic_series.append(np.nan)
                continue

            if method == "spearman":
                ic, _ = stats.spearmanr(pred[common], ret[common])
            else:
                ic, _ = stats.pearsonr(pred[common], ret[common])

            ic_series.append(ic)

        return pd.Series(ic_series, index=predictions.index)


def compute_icir(
    daily_ic: pd.Series,
    annualize: bool = True,
) -> float:
    """
    Compute Information Coefficient Information Ratio.

    ICIR = mean(IC) / std(IC) * sqrt(252) for annualized

    Args:
        daily_ic: Series of daily IC values
        annualize: Whether to annualize (multiply by sqrt(252))

    Returns:
        ICIR value
    """
    ic_clean = daily_ic.dropna()

    if len(ic_clean) < 10:
        return np.nan

    mean_ic = ic_clean.mean()
    std_ic = ic_clean.std()

    if std_ic == 0:
        return np.nan

    icir = mean_ic / std_ic

    if annualize:
        icir *= np.sqrt(252)

    return icir


def compute_sharpe(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> float:
    """
    Compute Sharpe Ratio.

    Args:
        returns: Daily return series
        risk_free_rate: Annual risk-free rate
        annualize: Whether to annualize

    Returns:
        Sharpe ratio
    """
    returns = returns.dropna()

    if len(returns) < 10:
        return np.nan

    excess_returns = returns - risk_free_rate / 252

    mean_return = excess_returns.mean()
    std_return = returns.std()

    if std_return == 0:
        return np.nan

    sharpe = mean_return / std_return

    if annualize:
        sharpe *= np.sqrt(252)

    return sharpe


def compute_max_drawdown(
    returns: pd.Series,
) -> float:
    """
    Compute maximum drawdown.

    Args:
        returns: Daily return series

    Returns:
        Maximum drawdown (negative value)
    """
    returns = returns.dropna()

    if len(returns) < 2:
        return 0.0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = cumulative / running_max - 1

    return drawdown.min()


def compute_turnover(
    weights: pd.DataFrame,
) -> pd.Series:
    """
    Compute daily portfolio turnover.

    Turnover = sum(|w_t - w_{t-1}|) / 2

    Args:
        weights: DataFrame of portfolio weights (dates x tickers)

    Returns:
        Series of daily turnover values
    """
    # Fill NaN with 0 (no position)
    weights = weights.fillna(0)

    # Compute absolute changes
    weight_changes = weights.diff().abs()

    # Sum across tickers, divide by 2 (one-way turnover)
    daily_turnover = weight_changes.sum(axis=1) / 2

    return daily_turnover


def compute_all_metrics(
    predictions: pd.DataFrame,
    forward_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    weights: pd.DataFrame,
    cost_bps: float = 7.5,
) -> PerformanceMetrics:
    """
    Compute all performance metrics.

    Args:
        predictions: Model predictions
        forward_returns: Actual forward returns
        portfolio_returns: Realized portfolio returns
        weights: Portfolio weights over time
        cost_bps: Transaction cost in bps

    Returns:
        PerformanceMetrics object
    """
    # IC metrics
    daily_ic = compute_daily_ic(predictions, forward_returns)
    mean_ic = daily_ic.mean()
    icir = compute_icir(daily_ic)
    ic_pvalue = stats.ttest_1samp(daily_ic.dropna(), 0).pvalue
    ic_hit_rate = (daily_ic > 0).mean()

    # Turnover
    daily_turnover = compute_turnover(weights)
    mean_turnover = daily_turnover.mean()

    # Cost-adjusted returns
    turnover_cost = mean_turnover * cost_bps / 10000
    net_returns = portfolio_returns - daily_turnover * cost_bps / 10000

    # Return metrics
    gross_sharpe = compute_sharpe(portfolio_returns)
    net_sharpe = compute_sharpe(net_returns)

    total_return = (1 + portfolio_returns).prod() - 1
    n_years = len(portfolio_returns) / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Risk metrics
    max_dd = compute_max_drawdown(portfolio_returns)
    volatility = portfolio_returns.std() * np.sqrt(252)
    calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    win_rate = (portfolio_returns > 0).mean()

    return PerformanceMetrics(
        mean_ic=mean_ic,
        icir=icir,
        ic_pvalue=ic_pvalue,
        ic_hit_rate=ic_hit_rate,
        gross_sharpe=gross_sharpe,
        net_sharpe=net_sharpe,
        total_return=total_return,
        annualized_return=annualized_return,
        max_drawdown=max_dd,
        volatility=volatility,
        calmar_ratio=calmar,
        mean_turnover=mean_turnover,
        turnover_cost=turnover_cost,
        n_days=len(portfolio_returns),
        win_rate=win_rate,
    )


class MetricsCalculator:
    """
    Comprehensive metrics calculator for backtesting.
    """

    def __init__(
        self,
        cost_bps: float = 7.5,
        settings=None,
    ):
        self.settings = settings or get_settings()
        self.cost_bps = cost_bps or self.settings.backtest.fixed_cost_bps

    def ic_decay_analysis(
        self,
        predictions: pd.DataFrame,
        returns_dict: Dict[int, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Analyze IC decay over different forward horizons.

        Args:
            predictions: Model predictions
            returns_dict: Dict of horizon -> forward returns DataFrames

        Returns:
            DataFrame with IC at each horizon
        """
        results = []

        for horizon, returns in returns_dict.items():
            daily_ic = compute_daily_ic(predictions, returns)
            results.append({
                "horizon": horizon,
                "mean_ic": daily_ic.mean(),
                "icir": compute_icir(daily_ic),
                "ic_std": daily_ic.std(),
            })

        return pd.DataFrame(results)

    def quantile_analysis(
        self,
        predictions: pd.DataFrame,
        forward_returns: pd.DataFrame,
        n_quantiles: int = 10,
    ) -> pd.DataFrame:
        """
        Analyze returns by prediction quantile.

        Args:
            predictions: Model predictions
            forward_returns: Forward returns
            n_quantiles: Number of quantiles

        Returns:
            DataFrame with returns by quantile
        """
        results = []

        for dt in predictions.index.get_level_values("date").unique():
            try:
                pred_day = predictions.xs(dt, level="date")
                ret_day = forward_returns.xs(dt, level="date")

                common = pred_day.index.intersection(ret_day.index)
                if len(common) < n_quantiles:
                    continue

                pred = pred_day.loc[common]
                ret = ret_day.loc[common]

                # Assign quantiles
                quantiles = pd.qcut(pred, n_quantiles, labels=False, duplicates="drop")

                for q in range(n_quantiles):
                    q_mask = quantiles == q
                    results.append({
                        "date": dt,
                        "quantile": q + 1,
                        "mean_return": ret[q_mask].mean(),
                        "n_stocks": q_mask.sum(),
                    })

            except Exception:
                continue

        df = pd.DataFrame(results)

        if len(df) == 0:
            return pd.DataFrame()

        # Aggregate
        summary = df.groupby("quantile").agg({
            "mean_return": "mean",
            "n_stocks": "mean",
        })

        return summary

    def compute_long_short_spread(
        self,
        quantile_returns: pd.DataFrame,
    ) -> float:
        """
        Compute long-short spread from quantile analysis.

        Long top decile, short bottom decile.
        """
        if len(quantile_returns) < 2:
            return np.nan

        long_return = quantile_returns.loc[quantile_returns.index.max(), "mean_return"]
        short_return = quantile_returns.loc[quantile_returns.index.min(), "mean_return"]

        return long_return - short_return

    def sector_ic_analysis(
        self,
        predictions: pd.DataFrame,
        forward_returns: pd.DataFrame,
        sectors: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute IC by sector.

        Args:
            predictions: Model predictions
            forward_returns: Forward returns
            sectors: Series mapping ticker to sector

        Returns:
            DataFrame with IC by sector
        """
        results = []

        for sector in sectors.unique():
            sector_tickers = sectors[sectors == sector].index

            if isinstance(predictions.index, pd.MultiIndex):
                pred = predictions.loc[predictions.index.get_level_values("ticker").isin(sector_tickers)]
                ret = forward_returns.loc[forward_returns.index.get_level_values("ticker").isin(sector_tickers)]
            else:
                pred = predictions[sector_tickers.intersection(predictions.columns)]
                ret = forward_returns[sector_tickers.intersection(forward_returns.columns)]

            daily_ic = compute_daily_ic(pred, ret)

            results.append({
                "sector": sector,
                "mean_ic": daily_ic.mean(),
                "icir": compute_icir(daily_ic),
                "n_stocks": len(sector_tickers),
            })

        return pd.DataFrame(results)

    def rolling_metrics(
        self,
        portfolio_returns: pd.Series,
        window: int = 252,
    ) -> pd.DataFrame:
        """
        Compute rolling performance metrics.

        Args:
            portfolio_returns: Daily portfolio returns
            window: Rolling window size

        Returns:
            DataFrame with rolling metrics
        """
        results = {
            "rolling_return": portfolio_returns.rolling(window).mean() * 252,
            "rolling_vol": portfolio_returns.rolling(window).std() * np.sqrt(252),
            "rolling_sharpe": (
                portfolio_returns.rolling(window).mean() /
                portfolio_returns.rolling(window).std() * np.sqrt(252)
            ),
        }

        # Rolling max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.rolling(window).max()
        rolling_dd = cumulative / rolling_max - 1
        results["rolling_max_dd"] = rolling_dd.rolling(window).min()

        return pd.DataFrame(results)
