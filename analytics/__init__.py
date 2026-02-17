"""
Advanced Real-Time Performance Analytics System
=============================================

Comprehensive trading performance monitoring with:
- Real-time P&L tracking
- Advanced performance ratios (Sharpe, Sortino, Calmar)
- Rolling performance windows
- Drawdown monitoring with alerts
- Strategy attribution analysis
- Risk-adjusted returns and alpha/beta calculations
- Portfolio correlation matrix and diversification metrics
- Benchmark comparison (SPY, BTC, etc.)
- Trade distribution analysis
- Performance alerts
- APIs for external consumption

Author: Trading Bot System
Created: February 2026
"""

from .core.performance_tracker import PerformanceTracker
from .core.pnl_calculator import PnLCalculator
from .metrics.performance_ratios import PerformanceRatios
from .metrics.drawdown_monitor import DrawdownMonitor
from .metrics.trade_statistics import TradeStatistics
from .benchmarks.benchmark_analyzer import BenchmarkAnalyzer
from .reporting.performance_reporter import PerformanceReporter

__version__ = "1.0.0"
__all__ = [
    "PerformanceTracker",
    "PnLCalculator",
    "PerformanceRatios",
    "DrawdownMonitor",
    "TradeStatistics",
    "BenchmarkAnalyzer",
    "PerformanceReporter"
]