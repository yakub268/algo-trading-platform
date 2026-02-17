"""
Trade Statistics Calculator
==========================

Comprehensive trade analysis and statistics including:
- Win rate and profit factor calculations
- Average win/loss analysis
- Trade distribution statistics
- Holding period analysis
- Monthly/daily performance breakdown
- Strategy performance attribution
- Risk metrics per trade

Author: Trading Bot System
Created: February 2026
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import sqlite3
from collections import defaultdict, Counter
import json

logger = logging.getLogger(__name__)


@dataclass
class TradeMetrics:
    """Comprehensive trade metrics container"""
    # Basic metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L metrics
    total_pnl: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float

    # Risk metrics
    expectancy: float
    sharpe_ratio: float
    calmar_ratio: float
    max_consecutive_losses: int
    max_consecutive_wins: int

    # Duration metrics
    average_hold_time: float  # hours
    median_hold_time: float  # hours
    longest_trade: float  # hours
    shortest_trade: float  # hours

    # Distribution metrics
    win_loss_ratio: float
    profit_margin: float
    recovery_factor: float
    ulcer_index: float

    # Metadata
    calculation_date: datetime
    period_start: datetime
    period_end: datetime

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class StrategyMetrics:
    """Strategy-specific performance metrics"""
    strategy_name: str
    metrics: TradeMetrics
    trade_count: int
    allocation_pct: float
    contribution_pct: float


class TradeStatistics:
    """
    Comprehensive trade statistics calculator providing detailed analysis
    of trading performance across multiple dimensions.

    Features:
    - Complete trade performance analysis
    - Strategy attribution statistics
    - Time-based performance breakdown
    - Risk-adjusted metrics
    - Distribution analysis
    - Comparative analytics
    """

    def __init__(self, db_path: str = None):
        """
        Initialize trade statistics calculator.

        Args:
            db_path: Database path for trade data
        """
        self.db_path = db_path or "data/trade_statistics.db"

        # Cache for frequent calculations
        self._cache = {}
        self._cache_expiry = {}
        self._cache_duration = 300  # 5 minutes

        self._init_database()
        logger.info("Trade Statistics calculator initialized")

    def _init_database(self):
        """Initialize database for trade statistics storage"""
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Statistics snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS statistics_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    calculation_date TEXT,
                    period_start TEXT,
                    period_end TEXT,
                    total_trades INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    total_pnl REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    metrics_json TEXT
                )
            ''')

            # Strategy performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    calculation_date TEXT,
                    strategy_name TEXT,
                    trade_count INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    total_pnl REAL,
                    allocation_pct REAL,
                    contribution_pct REAL,
                    metrics_json TEXT
                )
            ''')

            conn.commit()

    def calculate_trade_metrics(
        self,
        trades_df: pd.DataFrame,
        benchmark_return: float = 0.02
    ) -> TradeMetrics:
        """
        Calculate comprehensive trade metrics from trades DataFrame.

        Args:
            trades_df: DataFrame with trade data
            benchmark_return: Annual benchmark return for Sharpe ratio

        Returns:
            TradeMetrics object with all calculated metrics
        """
        if trades_df.empty:
            return self._empty_metrics()

        # Ensure required columns exist
        required_columns = ['pnl', 'entry_time', 'exit_time']
        for col in required_columns:
            if col not in trades_df.columns:
                logger.error(f"Missing required column: {col}")
                return self._empty_metrics()

        # Filter completed trades only
        completed_trades = trades_df[trades_df['exit_time'].notna()].copy()

        if completed_trades.empty:
            return self._empty_metrics()

        # Calculate basic metrics
        total_trades = len(completed_trades)
        winning_trades = len(completed_trades[completed_trades['pnl'] > 0])
        losing_trades = len(completed_trades[completed_trades['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # P&L metrics
        total_pnl = completed_trades['pnl'].sum()
        gross_profit = completed_trades[completed_trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(completed_trades[completed_trades['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Win/Loss analysis
        winning_pnl = completed_trades[completed_trades['pnl'] > 0]['pnl']
        losing_pnl = completed_trades[completed_trades['pnl'] < 0]['pnl']

        average_win = winning_pnl.mean() if len(winning_pnl) > 0 else 0
        average_loss = losing_pnl.mean() if len(losing_pnl) > 0 else 0
        largest_win = winning_pnl.max() if len(winning_pnl) > 0 else 0
        largest_loss = losing_pnl.min() if len(losing_pnl) > 0 else 0

        # Duration analysis
        completed_trades['entry_time'] = pd.to_datetime(completed_trades['entry_time'])
        completed_trades['exit_time'] = pd.to_datetime(completed_trades['exit_time'])
        completed_trades['hold_time_hours'] = (
            completed_trades['exit_time'] - completed_trades['entry_time']
        ).dt.total_seconds() / 3600

        average_hold_time = completed_trades['hold_time_hours'].mean()
        median_hold_time = completed_trades['hold_time_hours'].median()
        longest_trade = completed_trades['hold_time_hours'].max()
        shortest_trade = completed_trades['hold_time_hours'].min()

        # Risk metrics
        expectancy = (win_rate * average_win) + ((1 - win_rate) * average_loss)

        returns = completed_trades['pnl'].values
        if len(returns) > 1:
            sharpe_ratio = self._calculate_trade_sharpe(returns, benchmark_return)
            calmar_ratio = self._calculate_trade_calmar(returns)
            ulcer_index = self._calculate_ulcer_index(returns)
        else:
            sharpe_ratio = 0.0
            calmar_ratio = 0.0
            ulcer_index = 0.0

        # Consecutive win/loss streaks
        consecutive_wins, consecutive_losses = self._calculate_streaks(completed_trades['pnl'])

        # Additional metrics
        win_loss_ratio = average_win / abs(average_loss) if average_loss != 0 else 0
        profit_margin = total_pnl / gross_profit if gross_profit > 0 else 0
        recovery_factor = total_pnl / abs(largest_loss) if largest_loss != 0 else 0

        # Create metrics object
        metrics = TradeMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            expectancy=expectancy,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            max_consecutive_losses=consecutive_losses,
            max_consecutive_wins=consecutive_wins,
            average_hold_time=average_hold_time,
            median_hold_time=median_hold_time,
            longest_trade=longest_trade,
            shortest_trade=shortest_trade,
            win_loss_ratio=win_loss_ratio,
            profit_margin=profit_margin,
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index,
            calculation_date=datetime.now(),
            period_start=completed_trades['entry_time'].min(),
            period_end=completed_trades['exit_time'].max()
        )

        return metrics

    def _calculate_trade_sharpe(self, returns: np.ndarray, benchmark_return: float) -> float:
        """Calculate Sharpe ratio for trades"""
        try:
            if len(returns) == 0:
                return 0.0

            # Convert to percentage returns
            pct_returns = returns / np.abs(returns).mean() if np.abs(returns).mean() > 0 else returns

            mean_return = np.mean(pct_returns)
            std_return = np.std(pct_returns)

            if std_return == 0:
                return 0.0

            # Annualize benchmark return for comparison
            daily_benchmark = benchmark_return / 252

            sharpe = (mean_return - daily_benchmark) / std_return
            return sharpe * np.sqrt(252)  # Annualize

        except Exception as e:
            logger.debug(f"Error calculating trade Sharpe ratio: {e}")
            return 0.0

    def _calculate_trade_calmar(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio for trades"""
        try:
            if len(returns) == 0:
                return 0.0

            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max)
            max_drawdown = np.min(drawdowns)

            if max_drawdown == 0:
                return 0.0

            total_return = cumulative_returns[-1]
            calmar = total_return / abs(max_drawdown)

            return calmar

        except Exception as e:
            logger.debug(f"Error calculating Calmar ratio: {e}")
            return 0.0

    def _calculate_ulcer_index(self, returns: np.ndarray) -> float:
        """Calculate Ulcer Index (downside risk measure)"""
        try:
            if len(returns) == 0:
                return 0.0

            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max

            # Ulcer Index = sqrt(mean(drawdown^2))
            ulcer = np.sqrt(np.mean(drawdowns ** 2))

            return ulcer

        except Exception as e:
            logger.debug(f"Error calculating Ulcer Index: {e}")
            return 0.0

    def _calculate_streaks(self, pnl_series: pd.Series) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses"""
        try:
            if len(pnl_series) == 0:
                return 0, 0

            # Convert to win/loss sequence
            outcomes = (pnl_series > 0).astype(int)  # 1 for win, 0 for loss

            max_wins = 0
            max_losses = 0
            current_wins = 0
            current_losses = 0

            for outcome in outcomes:
                if outcome == 1:  # Win
                    current_wins += 1
                    current_losses = 0
                    max_wins = max(max_wins, current_wins)
                else:  # Loss
                    current_losses += 1
                    current_wins = 0
                    max_losses = max(max_losses, current_losses)

            return max_wins, max_losses

        except Exception as e:
            logger.debug(f"Error calculating streaks: {e}")
            return 0, 0

    def analyze_by_strategy(
        self,
        trades_df: pd.DataFrame,
        portfolio_value: float = 10000
    ) -> Dict[str, StrategyMetrics]:
        """
        Analyze performance by strategy.

        Args:
            trades_df: DataFrame with trade data including strategy column
            portfolio_value: Current portfolio value for allocation calculation

        Returns:
            Dictionary of strategy metrics
        """
        if trades_df.empty or 'strategy' not in trades_df.columns:
            return {}

        strategy_metrics = {}
        total_pnl = trades_df['pnl'].sum()

        for strategy, strategy_trades in trades_df.groupby('strategy'):
            if len(strategy_trades) == 0:
                continue

            # Calculate metrics for this strategy
            metrics = self.calculate_trade_metrics(strategy_trades)

            # Calculate allocation and contribution
            strategy_pnl = strategy_trades['pnl'].sum()
            allocation_pct = len(strategy_trades) / len(trades_df)
            contribution_pct = strategy_pnl / total_pnl if total_pnl != 0 else 0

            strategy_metrics[strategy] = StrategyMetrics(
                strategy_name=strategy,
                metrics=metrics,
                trade_count=len(strategy_trades),
                allocation_pct=allocation_pct,
                contribution_pct=contribution_pct
            )

        return strategy_metrics

    def analyze_by_time_period(
        self,
        trades_df: pd.DataFrame,
        period: str = 'M'  # 'D' for daily, 'W' for weekly, 'M' for monthly
    ) -> pd.DataFrame:
        """
        Analyze performance by time periods.

        Args:
            trades_df: DataFrame with trade data
            period: Time period ('D', 'W', 'M', 'Q', 'Y')

        Returns:
            DataFrame with time-based performance metrics
        """
        if trades_df.empty:
            return pd.DataFrame()

        # Ensure datetime columns
        trades_df = trades_df.copy()
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

        # Filter completed trades
        completed_trades = trades_df[trades_df['exit_time'].notna()].copy()

        if completed_trades.empty:
            return pd.DataFrame()

        # Group by time period
        completed_trades.set_index('exit_time', inplace=True)
        grouped = completed_trades.groupby(pd.Grouper(freq=period))

        results = []
        for period_date, period_trades in grouped:
            if len(period_trades) == 0:
                continue

            # Calculate basic metrics for the period
            period_result = {
                'period': period_date,
                'trades': len(period_trades),
                'pnl': period_trades['pnl'].sum(),
                'wins': len(period_trades[period_trades['pnl'] > 0]),
                'losses': len(period_trades[period_trades['pnl'] < 0]),
                'win_rate': len(period_trades[period_trades['pnl'] > 0]) / len(period_trades),
                'avg_pnl': period_trades['pnl'].mean(),
                'best_trade': period_trades['pnl'].max(),
                'worst_trade': period_trades['pnl'].min(),
                'profit_factor': (
                    period_trades[period_trades['pnl'] > 0]['pnl'].sum() /
                    abs(period_trades[period_trades['pnl'] < 0]['pnl'].sum())
                    if len(period_trades[period_trades['pnl'] < 0]) > 0 else float('inf')
                )
            }
            results.append(period_result)

        return pd.DataFrame(results)

    def get_trade_distribution(self, trades_df: pd.DataFrame) -> Dict:
        """
        Analyze trade P&L distribution.

        Args:
            trades_df: DataFrame with trade data

        Returns:
            Dictionary with distribution statistics
        """
        if trades_df.empty:
            return {}

        pnl_data = trades_df[trades_df['pnl'].notna()]['pnl']

        if len(pnl_data) == 0:
            return {}

        # Calculate distribution statistics
        distribution = {
            'mean': pnl_data.mean(),
            'median': pnl_data.median(),
            'std': pnl_data.std(),
            'skewness': pnl_data.skew(),
            'kurtosis': pnl_data.kurtosis(),
            'min': pnl_data.min(),
            'max': pnl_data.max(),
            'range': pnl_data.max() - pnl_data.min(),
            'percentiles': {
                'p5': pnl_data.quantile(0.05),
                'p25': pnl_data.quantile(0.25),
                'p75': pnl_data.quantile(0.75),
                'p95': pnl_data.quantile(0.95)
            },
            'bins': self._create_pnl_bins(pnl_data)
        }

        return distribution

    def _create_pnl_bins(self, pnl_data: pd.Series, n_bins: int = 10) -> Dict:
        """Create P&L distribution bins"""
        try:
            counts, bin_edges = np.histogram(pnl_data, bins=n_bins)

            bins = []
            for i in range(len(counts)):
                bins.append({
                    'range': f"${bin_edges[i]:.0f} to ${bin_edges[i+1]:.0f}",
                    'count': int(counts[i]),
                    'percentage': float(counts[i] / len(pnl_data) * 100)
                })

            return {'bins': bins, 'total_trades': len(pnl_data)}

        except Exception as e:
            logger.debug(f"Error creating P&L bins: {e}")
            return {}

    def get_monthly_performance(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Get monthly performance breakdown"""
        monthly_data = self.analyze_by_time_period(trades_df, 'M')

        if monthly_data.empty:
            return monthly_data

        # Add cumulative metrics
        monthly_data['cumulative_pnl'] = monthly_data['pnl'].cumsum()
        monthly_data['month'] = monthly_data['period'].dt.strftime('%Y-%m')

        return monthly_data

    def get_performance_summary(self, trades_df: pd.DataFrame) -> Dict:
        """Get comprehensive performance summary"""
        if trades_df.empty:
            return {}

        # Calculate main metrics
        main_metrics = self.calculate_trade_metrics(trades_df)

        # Strategy breakdown
        strategy_metrics = self.analyze_by_strategy(trades_df)

        # Distribution analysis
        distribution = self.get_trade_distribution(trades_df)

        # Monthly performance
        monthly = self.get_monthly_performance(trades_df)

        return {
            'overall_metrics': main_metrics.to_dict(),
            'strategy_breakdown': {
                name: {
                    'metrics': metrics.metrics.to_dict(),
                    'allocation_pct': metrics.allocation_pct,
                    'contribution_pct': metrics.contribution_pct,
                    'trade_count': metrics.trade_count
                }
                for name, metrics in strategy_metrics.items()
            },
            'distribution_analysis': distribution,
            'monthly_performance': monthly.to_dict('records') if not monthly.empty else [],
            'summary_stats': {
                'total_trades': main_metrics.total_trades,
                'win_rate': main_metrics.win_rate,
                'profit_factor': main_metrics.profit_factor,
                'sharpe_ratio': main_metrics.sharpe_ratio,
                'max_drawdown': distribution.get('min', 0),
                'best_month': monthly['pnl'].max() if not monthly.empty else 0,
                'worst_month': monthly['pnl'].min() if not monthly.empty else 0
            }
        }

    def _empty_metrics(self) -> TradeMetrics:
        """Return empty metrics object"""
        return TradeMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            profit_factor=0.0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            expectancy=0.0,
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            max_consecutive_losses=0,
            max_consecutive_wins=0,
            average_hold_time=0.0,
            median_hold_time=0.0,
            longest_trade=0.0,
            shortest_trade=0.0,
            win_loss_ratio=0.0,
            profit_margin=0.0,
            recovery_factor=0.0,
            ulcer_index=0.0,
            calculation_date=datetime.now(),
            period_start=datetime.now(),
            period_end=datetime.now()
        )

    def save_metrics(self, metrics: TradeMetrics, strategy_metrics: Dict[str, StrategyMetrics] = None):
        """Save calculated metrics to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Save overall metrics
                cursor.execute('''
                    INSERT INTO statistics_snapshots
                    (calculation_date, period_start, period_end, total_trades,
                     win_rate, profit_factor, total_pnl, sharpe_ratio,
                     max_drawdown, metrics_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.calculation_date.isoformat(),
                    metrics.period_start.isoformat(),
                    metrics.period_end.isoformat(),
                    metrics.total_trades,
                    metrics.win_rate,
                    metrics.profit_factor,
                    metrics.total_pnl,
                    metrics.sharpe_ratio,
                    metrics.largest_loss,
                    json.dumps(metrics.to_dict())
                ))

                # Save strategy metrics if provided
                if strategy_metrics:
                    for strategy_name, strategy_metric in strategy_metrics.items():
                        cursor.execute('''
                            INSERT INTO strategy_performance
                            (calculation_date, strategy_name, trade_count, win_rate,
                             profit_factor, total_pnl, allocation_pct, contribution_pct, metrics_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            metrics.calculation_date.isoformat(),
                            strategy_name,
                            strategy_metric.trade_count,
                            strategy_metric.metrics.win_rate,
                            strategy_metric.metrics.profit_factor,
                            strategy_metric.metrics.total_pnl,
                            strategy_metric.allocation_pct,
                            strategy_metric.contribution_pct,
                            json.dumps(strategy_metric.metrics.to_dict())
                        ))

                conn.commit()
                logger.info("Saved trade statistics to database")

        except Exception as e:
            logger.error(f"Error saving metrics: {e}")


# Example usage
if __name__ == "__main__":
    # Create sample trade data
    sample_trades = pd.DataFrame({
        'pnl': [100, -50, 200, -30, 150, -80, 300, -40, 90, -60],
        'entry_time': pd.date_range('2024-01-01', periods=10, freq='D'),
        'exit_time': pd.date_range('2024-01-02', periods=10, freq='D'),
        'strategy': ['momentum'] * 5 + ['mean_reversion'] * 5,
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'] * 2
    })

    # Initialize calculator
    calculator = TradeStatistics()

    # Calculate metrics
    metrics = calculator.calculate_trade_metrics(sample_trades)
    print(f"Win Rate: {metrics.win_rate:.1%}")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"Total P&L: ${metrics.total_pnl:.2f}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")

    # Strategy analysis
    strategy_metrics = calculator.analyze_by_strategy(sample_trades)
    for name, strat_metrics in strategy_metrics.items():
        print(f"\nStrategy: {name}")
        print(f"  Win Rate: {strat_metrics.metrics.win_rate:.1%}")
        print(f"  Contribution: {strat_metrics.contribution_pct:.1%}")

    # Get comprehensive summary
    summary = calculator.get_performance_summary(sample_trades)
    print(f"\nOverall Performance Summary:")
    print(f"Total Trades: {summary['summary_stats']['total_trades']}")
    print(f"Win Rate: {summary['summary_stats']['win_rate']:.1%}")
    print(f"Profit Factor: {summary['summary_stats']['profit_factor']:.2f}")