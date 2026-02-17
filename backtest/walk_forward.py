"""
Walk-Forward Testing Framework
Implements anchored walk-forward optimization with GO/NO-GO criteria.
"""

import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GoNoGoStatus(Enum):
    GO = "GO"
    NO_GO = "NO_GO"
    MARGINAL = "MARGINAL"


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward window."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # Training metrics
    train_trades: int
    train_pnl: float
    train_sharpe: float
    train_win_rate: float
    train_max_drawdown: float

    # Test (out-of-sample) metrics
    test_trades: int
    test_pnl: float
    test_sharpe: float
    test_win_rate: float
    test_max_drawdown: float

    # Best parameters found in training
    optimized_params: Dict[str, Any] = field(default_factory=dict)

    # Status
    go_nogo: GoNoGoStatus = GoNoGoStatus.NO_GO
    reason: str = ""


@dataclass
class WalkForwardSummary:
    """Summary of entire walk-forward test."""
    strategy_name: str
    total_windows: int
    go_windows: int
    nogo_windows: int
    marginal_windows: int

    # Aggregated out-of-sample metrics
    total_trades: int
    total_pnl: float
    avg_sharpe: float
    avg_win_rate: float
    worst_drawdown: float

    # Consistency metrics
    profit_factor: float
    oos_efficiency: float  # OOS performance / IS performance ratio

    # Results by window
    window_results: List[WalkForwardResult] = field(default_factory=list)

    # Final recommendation
    overall_status: GoNoGoStatus = GoNoGoStatus.NO_GO
    recommendation: str = ""


# GO/NO-GO Criteria thresholds
class GoNoGoCriteria:
    """Criteria for GO/NO-GO assessment."""
    MIN_SHARPE = 1.0
    MIN_TRADES = 100
    MAX_DRAWDOWN = -15.0  # V4: -15%
    MIN_WIN_RATE = 45.0  # 45%

    # Marginal thresholds (between GO and NO_GO)
    MARGINAL_SHARPE = 0.7
    MARGINAL_WIN_RATE = 40.0
    MARGINAL_DRAWDOWN = -20.0  # V4: tightened from -25%


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
    return float(sharpe)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown percentage."""
    if len(equity_curve) < 2:
        return 0.0

    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100
    return float(drawdown.min())


def calculate_profit_factor(trades: pd.DataFrame) -> float:
    """Calculate profit factor (gross profits / gross losses)."""
    if 'pnl' not in trades.columns or len(trades) == 0:
        return 0.0

    gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def assess_go_nogo(
    sharpe: float,
    trades: int,
    max_drawdown: float,
    win_rate: float
) -> Tuple[GoNoGoStatus, str]:
    """
    Assess GO/NO-GO status based on criteria.

    Returns:
        Tuple of (status, reason)
    """
    reasons = []

    # Check hard failures (NO_GO)
    if trades < GoNoGoCriteria.MIN_TRADES:
        reasons.append(f"Insufficient trades ({trades} < {GoNoGoCriteria.MIN_TRADES})")
    if max_drawdown < GoNoGoCriteria.MAX_DRAWDOWN:
        reasons.append(f"Excessive drawdown ({max_drawdown:.1f}% < {GoNoGoCriteria.MAX_DRAWDOWN}%)")

    if reasons:
        return GoNoGoStatus.NO_GO, "; ".join(reasons)

    # Check for GO status
    meets_sharpe = sharpe >= GoNoGoCriteria.MIN_SHARPE
    meets_win_rate = win_rate >= GoNoGoCriteria.MIN_WIN_RATE
    meets_drawdown = max_drawdown >= GoNoGoCriteria.MAX_DRAWDOWN

    if meets_sharpe and meets_win_rate and meets_drawdown:
        return GoNoGoStatus.GO, f"All criteria met (Sharpe={sharpe:.2f}, WR={win_rate:.1f}%, DD={max_drawdown:.1f}%)"

    # Check for MARGINAL status
    marginal_sharpe = sharpe >= GoNoGoCriteria.MARGINAL_SHARPE
    marginal_win_rate = win_rate >= GoNoGoCriteria.MARGINAL_WIN_RATE
    marginal_drawdown = max_drawdown >= GoNoGoCriteria.MARGINAL_DRAWDOWN

    if marginal_sharpe and marginal_win_rate and marginal_drawdown:
        warnings = []
        if not meets_sharpe:
            warnings.append(f"Sharpe below target ({sharpe:.2f} < {GoNoGoCriteria.MIN_SHARPE})")
        if not meets_win_rate:
            warnings.append(f"Win rate below target ({win_rate:.1f}% < {GoNoGoCriteria.MIN_WIN_RATE}%)")
        return GoNoGoStatus.MARGINAL, "; ".join(warnings)

    # NO_GO
    failures = []
    if not marginal_sharpe:
        failures.append(f"Low Sharpe ({sharpe:.2f})")
    if not marginal_win_rate:
        failures.append(f"Low win rate ({win_rate:.1f}%)")
    if not marginal_drawdown:
        failures.append(f"High drawdown ({max_drawdown:.1f}%)")

    return GoNoGoStatus.NO_GO, "; ".join(failures)


def walk_forward_test(
    data: pd.DataFrame,
    strategy_func: Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame],
    optimize_func: Callable[[pd.DataFrame], Dict[str, Any]],
    train_window: int = 252,  # 1 year of trading days
    test_window: int = 63,    # 3 months of trading days
    min_train_window: int = 126,  # Minimum 6 months for first window
    anchored: bool = True,
    strategy_name: str = "Strategy"
) -> WalkForwardSummary:
    """
    Perform walk-forward analysis on a trading strategy.

    Args:
        data: DataFrame with OHLCV data (must have 'close' column and datetime index)
        strategy_func: Function that takes (data, params) and returns DataFrame with 'pnl' column
        optimize_func: Function that takes training data and returns optimized parameters
        train_window: Number of bars for training window
        test_window: Number of bars for test window
        min_train_window: Minimum training bars for first window
        anchored: If True, training window grows (anchored); if False, rolling window
        strategy_name: Name of the strategy for reporting

    Returns:
        WalkForwardSummary with all results
    """
    logger.info(f"Starting walk-forward test for {strategy_name}")
    logger.info(f"Data range: {data.index[0]} to {data.index[-1]} ({len(data)} bars)")

    window_results = []
    window_id = 0

    # Determine window start positions
    current_pos = min_train_window

    while current_pos + test_window <= len(data):
        window_id += 1

        # Define window boundaries
        if anchored:
            train_start_idx = 0
        else:
            train_start_idx = max(0, current_pos - train_window)

        train_end_idx = current_pos
        test_start_idx = current_pos
        test_end_idx = min(current_pos + test_window, len(data))

        # Extract data slices
        train_data = data.iloc[train_start_idx:train_end_idx].copy()
        test_data = data.iloc[test_start_idx:test_end_idx].copy()

        train_start = train_data.index[0]
        train_end = train_data.index[-1]
        test_start = test_data.index[0]
        test_end = test_data.index[-1]

        logger.info(f"Window {window_id}: Train {train_start} to {train_end}, Test {test_start} to {test_end}")

        # Optimize on training data
        try:
            optimized_params = optimize_func(train_data)
        except Exception as e:
            logger.error(f"Optimization failed for window {window_id}: {e}")
            optimized_params = {}

        # Run strategy on training data
        try:
            train_results = strategy_func(train_data, optimized_params)
            train_trades = len(train_results[train_results['pnl'] != 0]) if 'pnl' in train_results.columns else 0
            train_pnl = train_results['pnl'].sum() if 'pnl' in train_results.columns else 0
            train_returns = train_results['pnl'] / train_data['close'].iloc[0] if 'pnl' in train_results.columns else pd.Series([0])
            train_sharpe = calculate_sharpe_ratio(train_returns)
            train_win_rate = (len(train_results[train_results['pnl'] > 0]) / train_trades * 100) if train_trades > 0 else 0
            train_equity = train_results['pnl'].cumsum() + 10000 if 'pnl' in train_results.columns else pd.Series([10000])
            train_max_dd = calculate_max_drawdown(train_equity)
        except Exception as e:
            logger.error(f"Training evaluation failed for window {window_id}: {e}")
            train_trades, train_pnl, train_sharpe, train_win_rate, train_max_dd = 0, 0, 0, 0, 0

        # Run strategy on test data (out-of-sample)
        try:
            test_results = strategy_func(test_data, optimized_params)
            test_trades = len(test_results[test_results['pnl'] != 0]) if 'pnl' in test_results.columns else 0
            test_pnl = test_results['pnl'].sum() if 'pnl' in test_results.columns else 0
            test_returns = test_results['pnl'] / test_data['close'].iloc[0] if 'pnl' in test_results.columns else pd.Series([0])
            test_sharpe = calculate_sharpe_ratio(test_returns)
            test_win_rate = (len(test_results[test_results['pnl'] > 0]) / test_trades * 100) if test_trades > 0 else 0
            test_equity = test_results['pnl'].cumsum() + 10000 if 'pnl' in test_results.columns else pd.Series([10000])
            test_max_dd = calculate_max_drawdown(test_equity)
        except Exception as e:
            logger.error(f"Test evaluation failed for window {window_id}: {e}")
            test_trades, test_pnl, test_sharpe, test_win_rate, test_max_dd = 0, 0, 0, 0, 0

        # Assess GO/NO-GO for this window
        go_nogo, reason = assess_go_nogo(test_sharpe, test_trades, test_max_dd, test_win_rate)

        result = WalkForwardResult(
            window_id=window_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_trades=train_trades,
            train_pnl=train_pnl,
            train_sharpe=train_sharpe,
            train_win_rate=train_win_rate,
            train_max_drawdown=train_max_dd,
            test_trades=test_trades,
            test_pnl=test_pnl,
            test_sharpe=test_sharpe,
            test_win_rate=test_win_rate,
            test_max_drawdown=test_max_dd,
            optimized_params=optimized_params,
            go_nogo=go_nogo,
            reason=reason
        )

        window_results.append(result)
        logger.info(f"Window {window_id} result: {go_nogo.value} - {reason}")

        # Move to next window
        current_pos += test_window

    # Calculate aggregate metrics
    total_windows = len(window_results)
    if total_windows == 0:
        logger.warning("No windows completed - insufficient data")
        return WalkForwardSummary(
            strategy_name=strategy_name,
            total_windows=0,
            go_windows=0,
            nogo_windows=0,
            marginal_windows=0,
            total_trades=0,
            total_pnl=0,
            avg_sharpe=0,
            avg_win_rate=0,
            worst_drawdown=0,
            profit_factor=0,
            oos_efficiency=0,
            window_results=[],
            overall_status=GoNoGoStatus.NO_GO,
            recommendation="Insufficient data for walk-forward analysis"
        )

    go_windows = sum(1 for r in window_results if r.go_nogo == GoNoGoStatus.GO)
    nogo_windows = sum(1 for r in window_results if r.go_nogo == GoNoGoStatus.NO_GO)
    marginal_windows = sum(1 for r in window_results if r.go_nogo == GoNoGoStatus.MARGINAL)

    total_trades = sum(r.test_trades for r in window_results)
    total_pnl = sum(r.test_pnl for r in window_results)
    avg_sharpe = np.mean([r.test_sharpe for r in window_results])
    avg_win_rate = np.mean([r.test_win_rate for r in window_results])
    worst_drawdown = min(r.test_max_drawdown for r in window_results)

    # Calculate profit factor from all OOS results
    all_pnls = [r.test_pnl for r in window_results]
    gross_profit = sum(p for p in all_pnls if p > 0)
    gross_loss = abs(sum(p for p in all_pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)

    # OOS efficiency (how well does OOS match IS)
    total_train_pnl = sum(r.train_pnl for r in window_results)
    oos_efficiency = total_pnl / total_train_pnl if total_train_pnl > 0 else 0

    # Determine overall status
    go_ratio = go_windows / total_windows
    if go_ratio >= 0.7 and avg_sharpe >= GoNoGoCriteria.MIN_SHARPE:
        overall_status = GoNoGoStatus.GO
        recommendation = f"Strategy approved for live trading. {go_windows}/{total_windows} windows passed GO criteria."
    elif go_ratio >= 0.5 or (go_ratio + marginal_windows / total_windows) >= 0.7:
        overall_status = GoNoGoStatus.MARGINAL
        recommendation = f"Strategy shows promise but needs refinement. Consider paper trading first."
    else:
        overall_status = GoNoGoStatus.NO_GO
        recommendation = f"Strategy not recommended. Only {go_windows}/{total_windows} windows passed GO criteria."

    summary = WalkForwardSummary(
        strategy_name=strategy_name,
        total_windows=total_windows,
        go_windows=go_windows,
        nogo_windows=nogo_windows,
        marginal_windows=marginal_windows,
        total_trades=total_trades,
        total_pnl=total_pnl,
        avg_sharpe=avg_sharpe,
        avg_win_rate=avg_win_rate,
        worst_drawdown=worst_drawdown,
        profit_factor=profit_factor,
        oos_efficiency=oos_efficiency,
        window_results=window_results,
        overall_status=overall_status,
        recommendation=recommendation
    )

    logger.info(f"Walk-forward complete: {overall_status.value} - {recommendation}")
    return summary


def print_walk_forward_report(summary: WalkForwardSummary) -> str:
    """Generate a formatted report of walk-forward results."""
    report = []
    report.append("=" * 70)
    report.append(f"WALK-FORWARD ANALYSIS REPORT: {summary.strategy_name}")
    report.append("=" * 70)
    report.append("")

    # Overall Status
    status_emoji = "✅" if summary.overall_status == GoNoGoStatus.GO else ("⚠️" if summary.overall_status == GoNoGoStatus.MARGINAL else "❌")
    report.append(f"OVERALL STATUS: {status_emoji} {summary.overall_status.value}")
    report.append(f"Recommendation: {summary.recommendation}")
    report.append("")

    # Summary Statistics
    report.append("-" * 70)
    report.append("SUMMARY STATISTICS (Out-of-Sample)")
    report.append("-" * 70)
    report.append(f"Total Windows:      {summary.total_windows}")
    report.append(f"GO Windows:         {summary.go_windows} ({summary.go_windows/summary.total_windows*100:.1f}%)")
    report.append(f"MARGINAL Windows:   {summary.marginal_windows} ({summary.marginal_windows/summary.total_windows*100:.1f}%)")
    report.append(f"NO-GO Windows:      {summary.nogo_windows} ({summary.nogo_windows/summary.total_windows*100:.1f}%)")
    report.append("")
    report.append(f"Total OOS Trades:   {summary.total_trades}")
    report.append(f"Total OOS P&L:      ${summary.total_pnl:,.2f}")
    report.append(f"Avg Sharpe Ratio:   {summary.avg_sharpe:.2f}")
    report.append(f"Avg Win Rate:       {summary.avg_win_rate:.1f}%")
    report.append(f"Worst Drawdown:     {summary.worst_drawdown:.1f}%")
    report.append(f"Profit Factor:      {summary.profit_factor:.2f}")
    report.append(f"OOS Efficiency:     {summary.oos_efficiency:.1%}")
    report.append("")

    # Window Details
    report.append("-" * 70)
    report.append("WINDOW DETAILS")
    report.append("-" * 70)

    for r in summary.window_results:
        status = "✅" if r.go_nogo == GoNoGoStatus.GO else ("⚠️" if r.go_nogo == GoNoGoStatus.MARGINAL else "❌")
        report.append(f"\nWindow {r.window_id}: {status} {r.go_nogo.value}")
        report.append(f"  Train: {r.train_start.strftime('%Y-%m-%d')} to {r.train_end.strftime('%Y-%m-%d')}")
        report.append(f"  Test:  {r.test_start.strftime('%Y-%m-%d')} to {r.test_end.strftime('%Y-%m-%d')}")
        report.append(f"  OOS Metrics: Trades={r.test_trades}, P&L=${r.test_pnl:.2f}, Sharpe={r.test_sharpe:.2f}, WR={r.test_win_rate:.1f}%, DD={r.test_max_drawdown:.1f}%")
        if r.reason:
            report.append(f"  Reason: {r.reason}")

    report.append("")
    report.append("=" * 70)
    report.append("GO/NO-GO CRITERIA")
    report.append("=" * 70)
    report.append(f"  Minimum Sharpe:     {GoNoGoCriteria.MIN_SHARPE}")
    report.append(f"  Minimum Trades:     {GoNoGoCriteria.MIN_TRADES}")
    report.append(f"  Maximum Drawdown:   {GoNoGoCriteria.MAX_DRAWDOWN}%")
    report.append(f"  Minimum Win Rate:   {GoNoGoCriteria.MIN_WIN_RATE}%")

    return "\n".join(report)


if __name__ == "__main__":
    # Example usage with dummy data
    import yfinance as yf

    logging.basicConfig(level=logging.INFO)

    # Download sample data
    print("Downloading sample data...")
    data = yf.download("SPY", start="2020-01-01", end="2024-01-01", progress=False)
    data.columns = [c.lower() for c in data.columns]

    # Simple moving average crossover strategy for demonstration
    def simple_ma_strategy(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        fast = params.get('fast_period', 10)
        slow = params.get('slow_period', 30)

        result = df.copy()
        result['sma_fast'] = result['close'].rolling(fast).mean()
        result['sma_slow'] = result['close'].rolling(slow).mean()

        # Generate signals
        result['signal'] = 0
        result.loc[result['sma_fast'] > result['sma_slow'], 'signal'] = 1
        result.loc[result['sma_fast'] < result['sma_slow'], 'signal'] = -1

        # Calculate P&L (simplified)
        result['returns'] = result['close'].pct_change()
        result['pnl'] = result['signal'].shift(1) * result['returns'] * 10000  # $10k position

        return result.dropna()

    def optimize_ma(df: pd.DataFrame) -> Dict[str, Any]:
        # Simple grid search for best parameters
        best_sharpe = -float('inf')
        best_params = {'fast_period': 10, 'slow_period': 30}

        for fast in [5, 10, 15, 20]:
            for slow in [20, 30, 40, 50]:
                if fast >= slow:
                    continue
                result = simple_ma_strategy(df, {'fast_period': fast, 'slow_period': slow})
                sharpe = calculate_sharpe_ratio(result['pnl'] / 10000)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {'fast_period': fast, 'slow_period': slow}

        return best_params

    # Run walk-forward test
    print("\nRunning walk-forward analysis...")
    summary = walk_forward_test(
        data=data,
        strategy_func=simple_ma_strategy,
        optimize_func=optimize_ma,
        train_window=252,
        test_window=63,
        strategy_name="MA Crossover"
    )

    # Print report
    report = print_walk_forward_report(summary)
    print(report)
