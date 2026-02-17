"""
VectorBT Backtesting Engine

High-performance vectorized backtesting using VectorBT library.
Optimized for parameter sweeps across 1000+ combinations.

Features:
- Fast vectorized operations via NumPy/Numba
- Multi-asset backtesting
- Parameter optimization with grid search
- Monte Carlo simulation integration
- Comprehensive metrics calculation

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('VectorBTEngine')

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    logger.warning("VectorBT not installed. Run: pip install vectorbt")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not installed. Using pandas-based indicators.")


@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    avg_holding_period: float
    exposure_time: float
    parameters: Dict
    equity_curve: pd.Series
    trades: pd.DataFrame


@dataclass
class OptimizationResult:
    """Parameter optimization results"""
    best_params: Dict
    best_sharpe: float
    best_return: float
    all_results: pd.DataFrame
    heatmap_data: Optional[pd.DataFrame] = None


class VectorBTEngine:
    """
    High-performance backtesting engine using VectorBT.
    
    Supports:
    - RSI-2 Mean Reversion
    - MACD + RSI Combo
    - Bollinger Band strategies
    - Custom strategy functions
    
    Features:
    - Grid search parameter optimization
    - Walk-forward validation
    - Monte Carlo simulation
    - Multi-asset testing
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize VectorBT Engine.
        
        Args:
            initial_capital: Starting capital for backtests
        """
        self.initial_capital = initial_capital
        
        if not VBT_AVAILABLE:
            raise ImportError("VectorBT required. Install with: pip install vectorbt")
        
        # Configure VectorBT settings
        vbt.settings.portfolio['init_cash'] = initial_capital
        vbt.settings.portfolio['fees'] = 0.001  # 0.1% commission
        vbt.settings.portfolio['slippage'] = 0.001  # 0.1% slippage
        
        logger.info(f"VectorBTEngine initialized with ${initial_capital:,.0f} capital")
    
    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 2) -> pd.Series:
        """Calculate RSI indicator"""
        if TALIB_AVAILABLE:
            return pd.Series(talib.RSI(close.values, timeperiod=period), index=close.index)
        else:
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
            avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_sma(close: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return close.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(close: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return close.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        if TALIB_AVAILABLE:
            macd, signal_line, hist = talib.MACD(close.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return pd.Series(macd, index=close.index), pd.Series(signal_line, index=close.index), pd.Series(hist, index=close.index)
        else:
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def backtest_rsi2(
        self,
        data: pd.DataFrame,
        rsi_period: int = 2,
        oversold: int = 10,
        overbought: int = 90,
        trend_sma: int = 200,
        exit_sma: int = 5,
        stop_loss: float = 0.03
    ) -> BacktestResult:
        """
        Backtest RSI-2 Mean Reversion Strategy.
        
        Entry: RSI(2) < oversold AND price > SMA(200)
        Exit: RSI(2) > overbought OR price > SMA(5)
        
        Args:
            data: DataFrame with OHLCV columns
            rsi_period: RSI calculation period
            oversold: RSI oversold threshold
            overbought: RSI overbought threshold
            trend_sma: Long-term trend SMA period
            exit_sma: Exit SMA period
            stop_loss: Stop loss percentage
            
        Returns:
            BacktestResult with all metrics
        """
        close = data['Close'] if 'Close' in data.columns else data['close']
        
        # Calculate indicators
        rsi = self.calculate_rsi(close, rsi_period)
        sma_trend = self.calculate_sma(close, trend_sma)
        sma_exit = self.calculate_sma(close, exit_sma)
        
        # Generate signals
        # Entry: RSI oversold AND price above trend SMA
        entries = (rsi < oversold) & (close > sma_trend)
        
        # Exit: RSI overbought OR price above exit SMA
        exits = (rsi > overbought) | (close > sma_exit)
        
        # Run backtest with VectorBT
        portfolio = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            fees=0.001,
            slippage=0.001,
            sl_stop=stop_loss,
            freq='1D'
        )
        
        # Extract metrics
        stats = portfolio.stats()
        
        return BacktestResult(
            strategy_name="RSI-2 Mean Reversion",
            symbol=data.get('symbol', 'UNKNOWN'),
            start_date=close.index[0],
            end_date=close.index[-1],
            initial_capital=self.initial_capital,
            final_value=float(stats.get('End Value', self.initial_capital)),
            total_return=float(stats.get('Total Return [%]', 0)) / 100,
            cagr=float(stats.get('Annualized Return [%]', 0)) / 100,
            sharpe_ratio=float(stats.get('Sharpe Ratio', 0)),
            sortino_ratio=float(stats.get('Sortino Ratio', 0)),
            calmar_ratio=float(stats.get('Calmar Ratio', 0)),
            max_drawdown=float(stats.get('Max Drawdown [%]', 0)) / 100,
            win_rate=float(stats.get('Win Rate [%]', 0)) / 100,
            profit_factor=float(stats.get('Profit Factor', 0)),
            total_trades=int(stats.get('Total Trades', 0)),
            avg_trade_return=float(stats.get('Avg Winning Trade [%]', 0)) / 100 if stats.get('Win Rate [%]', 0) > 0 else 0,
            avg_win=float(stats.get('Avg Winning Trade [%]', 0)) / 100,
            avg_loss=float(stats.get('Avg Losing Trade [%]', 0)) / 100,
            best_trade=float(stats.get('Best Trade [%]', 0)) / 100,
            worst_trade=float(stats.get('Worst Trade [%]', 0)) / 100,
            avg_holding_period=float(stats.get('Avg Winning Trade Duration', pd.Timedelta(0)).days) if isinstance(stats.get('Avg Winning Trade Duration'), pd.Timedelta) else 0,
            exposure_time=float(stats.get('Exposure Time [%]', 0)) / 100,
            parameters={
                'rsi_period': rsi_period,
                'oversold': oversold,
                'overbought': overbought,
                'trend_sma': trend_sma,
                'exit_sma': exit_sma,
                'stop_loss': stop_loss
            },
            equity_curve=portfolio.value(),
            trades=portfolio.trades.records_readable if hasattr(portfolio.trades, 'records_readable') else pd.DataFrame()
        )
    
    def optimize_rsi2(
        self,
        data: pd.DataFrame,
        rsi_range: Tuple[int, int] = (2, 5),
        oversold_range: Tuple[int, int] = (5, 20),
        exit_sma_range: Tuple[int, int] = (3, 10),
        metric: str = 'sharpe'
    ) -> OptimizationResult:
        """
        Optimize RSI-2 strategy parameters using grid search.
        
        Args:
            data: DataFrame with OHLCV data
            rsi_range: Range for RSI period (min, max)
            oversold_range: Range for oversold threshold (min, max)
            exit_sma_range: Range for exit SMA (min, max)
            metric: Optimization metric ('sharpe', 'return', 'calmar')
            
        Returns:
            OptimizationResult with best parameters and all results
        """
        close = data['Close'] if 'Close' in data.columns else data['close']
        
        # Parameter grids
        rsi_periods = list(range(rsi_range[0], rsi_range[1] + 1))
        oversold_levels = list(range(oversold_range[0], oversold_range[1] + 1, 5))
        exit_smas = list(range(exit_sma_range[0], exit_sma_range[1] + 1))
        
        results = []
        best_sharpe = -np.inf
        best_params = {}
        
        total_combinations = len(rsi_periods) * len(oversold_levels) * len(exit_smas)
        logger.info(f"Running optimization with {total_combinations} parameter combinations...")
        
        for rsi_period in rsi_periods:
            for oversold in oversold_levels:
                for exit_sma in exit_smas:
                    try:
                        result = self.backtest_rsi2(
                            data,
                            rsi_period=rsi_period,
                            oversold=oversold,
                            exit_sma=exit_sma
                        )
                        
                        results.append({
                            'rsi_period': rsi_period,
                            'oversold': oversold,
                            'exit_sma': exit_sma,
                            'sharpe': result.sharpe_ratio,
                            'return': result.total_return,
                            'max_dd': result.max_drawdown,
                            'win_rate': result.win_rate,
                            'trades': result.total_trades
                        })
                        
                        if result.sharpe_ratio > best_sharpe and result.total_trades >= 10:
                            best_sharpe = result.sharpe_ratio
                            best_params = {
                                'rsi_period': rsi_period,
                                'oversold': oversold,
                                'exit_sma': exit_sma
                            }
                    except Exception as e:
                        logger.debug(f"Combination failed: {e}")
        
        results_df = pd.DataFrame(results)
        
        # Create heatmap data for RSI vs Oversold
        heatmap = results_df.pivot_table(
            values='sharpe',
            index='oversold',
            columns='rsi_period',
            aggfunc='mean'
        )
        
        logger.info(f"Optimization complete. Best Sharpe: {best_sharpe:.2f}")
        logger.info(f"Best params: {best_params}")
        
        return OptimizationResult(
            best_params=best_params,
            best_sharpe=best_sharpe,
            best_return=results_df.loc[results_df['sharpe'] == best_sharpe, 'return'].values[0] if len(results_df) > 0 else 0,
            all_results=results_df,
            heatmap_data=heatmap
        )
    
    def backtest_macd_rsi(
        self,
        data: pd.DataFrame,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rsi_period: int = 14,
        rsi_lower: int = 30,
        rsi_upper: int = 70,
        stop_loss: float = 0.03
    ) -> BacktestResult:
        """
        Backtest MACD + RSI Combined Strategy.
        
        Entry: MACD crosses above signal AND RSI between 30-50
        Exit: MACD crosses below signal OR RSI > 70
        
        Args:
            data: DataFrame with OHLCV columns
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
            rsi_period: RSI period
            rsi_lower: RSI lower bound for entry
            rsi_upper: RSI upper bound for exit
            stop_loss: Stop loss percentage
            
        Returns:
            BacktestResult with all metrics
        """
        close = data['Close'] if 'Close' in data.columns else data['close']
        
        # Calculate indicators
        macd_line, signal_line, _ = self.calculate_macd(close, macd_fast, macd_slow, macd_signal)
        rsi = self.calculate_rsi(close, rsi_period)
        sma50 = self.calculate_sma(close, 50)
        
        # Generate signals
        # MACD crossover
        macd_cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        macd_cross_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        
        # Entry: MACD cross up AND RSI in range AND above SMA50
        entries = macd_cross_up & (rsi > rsi_lower) & (rsi < 50) & (close > sma50)
        
        # Exit: MACD cross down OR RSI overbought
        exits = macd_cross_down | (rsi > rsi_upper)
        
        # Run backtest
        portfolio = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            fees=0.001,
            slippage=0.001,
            sl_stop=stop_loss,
            freq='1D'
        )
        
        stats = portfolio.stats()
        
        return BacktestResult(
            strategy_name="MACD + RSI Combo",
            symbol=data.get('symbol', 'UNKNOWN'),
            start_date=close.index[0],
            end_date=close.index[-1],
            initial_capital=self.initial_capital,
            final_value=float(stats.get('End Value', self.initial_capital)),
            total_return=float(stats.get('Total Return [%]', 0)) / 100,
            cagr=float(stats.get('Annualized Return [%]', 0)) / 100,
            sharpe_ratio=float(stats.get('Sharpe Ratio', 0)),
            sortino_ratio=float(stats.get('Sortino Ratio', 0)),
            calmar_ratio=float(stats.get('Calmar Ratio', 0)),
            max_drawdown=float(stats.get('Max Drawdown [%]', 0)) / 100,
            win_rate=float(stats.get('Win Rate [%]', 0)) / 100,
            profit_factor=float(stats.get('Profit Factor', 0)),
            total_trades=int(stats.get('Total Trades', 0)),
            avg_trade_return=float(stats.get('Avg Winning Trade [%]', 0)) / 100,
            avg_win=float(stats.get('Avg Winning Trade [%]', 0)) / 100,
            avg_loss=float(stats.get('Avg Losing Trade [%]', 0)) / 100,
            best_trade=float(stats.get('Best Trade [%]', 0)) / 100,
            worst_trade=float(stats.get('Worst Trade [%]', 0)) / 100,
            avg_holding_period=0,
            exposure_time=float(stats.get('Exposure Time [%]', 0)) / 100,
            parameters={
                'macd_fast': macd_fast,
                'macd_slow': macd_slow,
                'macd_signal': macd_signal,
                'rsi_period': rsi_period,
                'rsi_lower': rsi_lower,
                'rsi_upper': rsi_upper,
                'stop_loss': stop_loss
            },
            equity_curve=portfolio.value(),
            trades=portfolio.trades.records_readable if hasattr(portfolio.trades, 'records_readable') else pd.DataFrame()
        )
    
    def backtest_bollinger(
        self,
        data: pd.DataFrame,
        bb_period: int = 20,
        bb_std: float = 2.0,
        squeeze_percentile: float = 0.2,
        stop_loss: float = 0.03
    ) -> BacktestResult:
        """
        Backtest Bollinger Band Squeeze Strategy.
        
        Entry: BB width at squeeze (low volatility) AND breakout above upper band
        Exit: Price closes below middle band
        
        Args:
            data: DataFrame with OHLCV columns
            bb_period: Bollinger Band period
            bb_std: Bollinger Band standard deviations
            squeeze_percentile: Width percentile for squeeze
            stop_loss: Stop loss percentage
            
        Returns:
            BacktestResult with all metrics
        """
        close = data['Close'] if 'Close' in data.columns else data['close']
        
        # Calculate Bollinger Bands
        upper, middle, lower = self.calculate_bollinger_bands(close, bb_period, bb_std)
        
        # Calculate band width
        width = (upper - lower) / middle
        width_percentile = width.rolling(window=126).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        # Squeeze condition: width in bottom percentile
        squeeze = width_percentile < squeeze_percentile
        
        # Breakout: price crosses above upper band after squeeze
        breakout = (close > upper) & squeeze.shift(1)
        
        # Exit: price below middle band
        exit_signal = close < middle
        
        # Run backtest
        portfolio = vbt.Portfolio.from_signals(
            close,
            entries=breakout,
            exits=exit_signal,
            init_cash=self.initial_capital,
            fees=0.001,
            slippage=0.001,
            sl_stop=stop_loss,
            freq='1D'
        )
        
        stats = portfolio.stats()
        
        return BacktestResult(
            strategy_name="Bollinger Squeeze",
            symbol=data.get('symbol', 'UNKNOWN'),
            start_date=close.index[0],
            end_date=close.index[-1],
            initial_capital=self.initial_capital,
            final_value=float(stats.get('End Value', self.initial_capital)),
            total_return=float(stats.get('Total Return [%]', 0)) / 100,
            cagr=float(stats.get('Annualized Return [%]', 0)) / 100,
            sharpe_ratio=float(stats.get('Sharpe Ratio', 0)),
            sortino_ratio=float(stats.get('Sortino Ratio', 0)),
            calmar_ratio=float(stats.get('Calmar Ratio', 0)),
            max_drawdown=float(stats.get('Max Drawdown [%]', 0)) / 100,
            win_rate=float(stats.get('Win Rate [%]', 0)) / 100,
            profit_factor=float(stats.get('Profit Factor', 0)),
            total_trades=int(stats.get('Total Trades', 0)),
            avg_trade_return=float(stats.get('Avg Winning Trade [%]', 0)) / 100,
            avg_win=float(stats.get('Avg Winning Trade [%]', 0)) / 100,
            avg_loss=float(stats.get('Avg Losing Trade [%]', 0)) / 100,
            best_trade=float(stats.get('Best Trade [%]', 0)) / 100,
            worst_trade=float(stats.get('Worst Trade [%]', 0)) / 100,
            avg_holding_period=0,
            exposure_time=float(stats.get('Exposure Time [%]', 0)) / 100,
            parameters={
                'bb_period': bb_period,
                'bb_std': bb_std,
                'squeeze_percentile': squeeze_percentile,
                'stop_loss': stop_loss
            },
            equity_curve=portfolio.value(),
            trades=portfolio.trades.records_readable if hasattr(portfolio.trades, 'records_readable') else pd.DataFrame()
        )
    
    def compare_strategies(
        self,
        data: pd.DataFrame,
        strategies: List[str] = ['rsi2', 'macd_rsi', 'bollinger']
    ) -> pd.DataFrame:
        """
        Compare multiple strategies on same data.
        
        Args:
            data: DataFrame with OHLCV data
            strategies: List of strategy names to compare
            
        Returns:
            DataFrame with comparison metrics
        """
        results = []
        
        for strategy in strategies:
            if strategy == 'rsi2':
                result = self.backtest_rsi2(data)
            elif strategy == 'macd_rsi':
                result = self.backtest_macd_rsi(data)
            elif strategy == 'bollinger':
                result = self.backtest_bollinger(data)
            else:
                logger.warning(f"Unknown strategy: {strategy}")
                continue
            
            results.append({
                'Strategy': result.strategy_name,
                'Total Return': f"{result.total_return:.1%}",
                'CAGR': f"{result.cagr:.1%}",
                'Sharpe': f"{result.sharpe_ratio:.2f}",
                'Sortino': f"{result.sortino_ratio:.2f}",
                'Max DD': f"{result.max_drawdown:.1%}",
                'Win Rate': f"{result.win_rate:.1%}",
                'Trades': result.total_trades,
                'Profit Factor': f"{result.profit_factor:.2f}"
            })
        
        return pd.DataFrame(results)
    
    def print_result(self, result: BacktestResult):
        """Pretty print backtest result"""
        print("\n" + "=" * 60)
        print(f"BACKTEST RESULT: {result.strategy_name}")
        print("=" * 60)
        print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
        print(f"Initial Capital: ${result.initial_capital:,.0f}")
        print(f"Final Value: ${result.final_value:,.0f}")
        print("-" * 60)
        print(f"Total Return: {result.total_return:+.1%}")
        print(f"CAGR: {result.cagr:+.1%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {result.sortino_ratio:.2f}")
        print(f"Max Drawdown: {result.max_drawdown:.1%}")
        print("-" * 60)
        print(f"Total Trades: {result.total_trades}")
        print(f"Win Rate: {result.win_rate:.1%}")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print(f"Avg Win: {result.avg_win:+.2%}")
        print(f"Avg Loss: {result.avg_loss:.2%}")
        print(f"Best Trade: {result.best_trade:+.2%}")
        print(f"Worst Trade: {result.worst_trade:.2%}")
        print("=" * 60)


# Example usage and testing
if __name__ == "__main__":
    import yfinance as yf
    
    print("=" * 60)
    print("VECTORBT BACKTESTING ENGINE")
    print("=" * 60)
    
    # Download test data
    print("\nDownloading SPY data...")
    spy = yf.download("SPY", period="5y", interval="1d", progress=False)
    
    if len(spy) > 0:
        print(f"Loaded {len(spy)} days of data")
        
        # Initialize engine
        engine = VectorBTEngine(initial_capital=10000)
        
        # Test RSI-2 strategy
        print("\n[1] Testing RSI-2 Strategy...")
        rsi2_result = engine.backtest_rsi2(spy)
        engine.print_result(rsi2_result)
        
        # Test MACD + RSI strategy
        print("\n[2] Testing MACD + RSI Strategy...")
        macd_result = engine.backtest_macd_rsi(spy)
        engine.print_result(macd_result)
        
        # Test Bollinger strategy
        print("\n[3] Testing Bollinger Squeeze Strategy...")
        bb_result = engine.backtest_bollinger(spy)
        engine.print_result(bb_result)
        
        # Compare all strategies
        print("\n[4] Strategy Comparison...")
        comparison = engine.compare_strategies(spy)
        print(comparison.to_string(index=False))
        
        # Optimize RSI-2
        print("\n[5] Optimizing RSI-2 Parameters...")
        opt_result = engine.optimize_rsi2(spy)
        print(f"\nBest Parameters: {opt_result.best_params}")
        print(f"Best Sharpe: {opt_result.best_sharpe:.2f}")
        print(f"Best Return: {opt_result.best_return:.1%}")
        
    else:
        print("Failed to load data. Install yfinance: pip install yfinance")
