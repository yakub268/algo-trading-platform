"""
Benchmark Analyzer - Comprehensive Benchmark Comparison System
============================================================

Advanced benchmark analysis including:
- Multiple benchmark comparisons (SPY, BTC, QQQ, etc.)
- Risk-adjusted performance comparisons
- Alpha and Beta calculations
- Correlation analysis
- Rolling comparisons
- Relative performance tracking
- Benchmark-relative drawdowns

Author: Trading Bot System
Created: February 2026
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import yfinance as yf
import sqlite3
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkComparison:
    """Results of benchmark comparison analysis"""
    benchmark_symbol: str
    benchmark_name: str
    period_days: int

    # Return metrics
    portfolio_return: float
    benchmark_return: float
    excess_return: float

    # Risk metrics
    portfolio_volatility: float
    benchmark_volatility: float
    tracking_error: float

    # Risk-adjusted metrics
    alpha: float
    beta: float
    correlation: float
    information_ratio: float
    treynor_ratio: float

    # Performance attribution
    selection_return: float  # Stock selection effect
    timing_return: float     # Market timing effect

    # Drawdown comparison
    max_portfolio_drawdown: float
    max_benchmark_drawdown: float
    relative_drawdown: float

    # Statistical significance
    alpha_p_value: float
    beta_p_value: float

    calculation_date: datetime

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class MultiBenchmarkSummary:
    """Summary of comparisons against multiple benchmarks"""
    best_alpha_benchmark: str
    best_alpha_value: float
    lowest_correlation_benchmark: str
    lowest_correlation_value: float
    best_information_ratio_benchmark: str
    best_information_ratio_value: float
    average_alpha: float
    average_beta: float
    average_correlation: float
    calculation_date: datetime


class BenchmarkAnalyzer:
    """
    Comprehensive benchmark analysis system for trading performance evaluation.

    Features:
    - Multi-benchmark comparison
    - Risk-adjusted performance metrics
    - Rolling benchmark analysis
    - Alpha/Beta decomposition
    - Performance attribution
    - Statistical significance testing
    - Benchmark data caching
    """

    def __init__(self, cache_duration_hours: int = 6):
        """
        Initialize benchmark analyzer.

        Args:
            cache_duration_hours: How long to cache benchmark data
        """
        self.cache_duration_hours = cache_duration_hours
        self.benchmark_cache: Dict[str, Dict] = {}
        self._cache_lock = threading.RLock()

        # Predefined benchmark configurations
        self.benchmark_configs = {
            'SPY': {'name': 'S&P 500 ETF', 'type': 'equity', 'region': 'US'},
            'QQQ': {'name': 'NASDAQ 100 ETF', 'type': 'equity_tech', 'region': 'US'},
            'IWM': {'name': 'Russell 2000 ETF', 'type': 'small_cap', 'region': 'US'},
            'VTI': {'name': 'Total Stock Market ETF', 'type': 'broad_market', 'region': 'US'},
            'EFA': {'name': 'MSCI EAFE ETF', 'type': 'international', 'region': 'Developed'},
            'EEM': {'name': 'Emerging Markets ETF', 'type': 'emerging', 'region': 'EM'},
            'TLT': {'name': '20+ Year Treasury ETF', 'type': 'bonds', 'region': 'US'},
            'GLD': {'name': 'Gold ETF', 'type': 'commodity', 'region': 'Global'},
            'BTC-USD': {'name': 'Bitcoin', 'type': 'crypto', 'region': 'Global'},
            'ETH-USD': {'name': 'Ethereum', 'type': 'crypto', 'region': 'Global'},
            '^VIX': {'name': 'VIX Index', 'type': 'volatility', 'region': 'US'},
            'DXY': {'name': 'US Dollar Index', 'type': 'currency', 'region': 'US'}
        }

        logger.info(f"Benchmark Analyzer initialized with {len(self.benchmark_configs)} benchmarks")

    def compare_to_benchmark(
        self,
        portfolio_returns: Union[pd.Series, List[float]],
        benchmark_symbol: str = 'SPY',
        portfolio_timestamps: Optional[pd.DatetimeIndex] = None
    ) -> BenchmarkComparison:
        """
        Compare portfolio performance to a single benchmark.

        Args:
            portfolio_returns: Portfolio returns (daily)
            benchmark_symbol: Benchmark ticker symbol
            portfolio_timestamps: Timestamps for portfolio returns

        Returns:
            BenchmarkComparison object with all metrics
        """
        try:
            # Convert to pandas Series if needed
            if isinstance(portfolio_returns, list):
                if portfolio_timestamps is None:
                    portfolio_timestamps = pd.date_range(
                        end=datetime.now(), periods=len(portfolio_returns), freq='D'
                    )
                portfolio_returns = pd.Series(portfolio_returns, index=portfolio_timestamps)
            elif not hasattr(portfolio_returns, 'index'):
                if portfolio_timestamps is None:
                    portfolio_timestamps = pd.date_range(
                        end=datetime.now(), periods=len(portfolio_returns), freq='D'
                    )
                portfolio_returns = pd.Series(portfolio_returns, index=portfolio_timestamps)

            # Get benchmark data
            benchmark_data = self._get_benchmark_data(
                benchmark_symbol,
                start_date=portfolio_returns.index[0],
                end_date=portfolio_returns.index[-1]
            )

            if benchmark_data.empty:
                return self._empty_comparison(benchmark_symbol)

            # Align data
            aligned_port, aligned_bench = self._align_returns(portfolio_returns, benchmark_data)

            if len(aligned_port) < 10:  # Need minimum observations
                return self._empty_comparison(benchmark_symbol)

            # Calculate basic metrics
            portfolio_return = aligned_port.sum()
            benchmark_return = aligned_bench.sum()
            excess_return = portfolio_return - benchmark_return

            # Risk metrics
            portfolio_vol = aligned_port.std() * np.sqrt(252)
            benchmark_vol = aligned_bench.std() * np.sqrt(252)
            tracking_error = (aligned_port - aligned_bench).std() * np.sqrt(252)

            # Alpha and Beta calculation
            alpha, beta, alpha_pval, beta_pval = self._calculate_alpha_beta_stats(
                aligned_port, aligned_bench
            )

            # Other metrics
            correlation = np.corrcoef(aligned_port, aligned_bench)[0, 1]

            # Information ratio
            if tracking_error > 0:
                information_ratio = excess_return / tracking_error * np.sqrt(252)
            else:
                information_ratio = 0.0

            # Treynor ratio (assuming risk-free rate = 0)
            treynor_ratio = portfolio_return / beta if beta != 0 else 0.0

            # Performance attribution
            selection_return, timing_return = self._calculate_attribution(
                aligned_port, aligned_bench
            )

            # Drawdown comparison
            portfolio_dd = self._calculate_max_drawdown(aligned_port)
            benchmark_dd = self._calculate_max_drawdown(aligned_bench)
            relative_dd = self._calculate_max_drawdown(aligned_port - aligned_bench)

            return BenchmarkComparison(
                benchmark_symbol=benchmark_symbol,
                benchmark_name=self.benchmark_configs.get(benchmark_symbol, {}).get('name', benchmark_symbol),
                period_days=len(aligned_port),
                portfolio_return=portfolio_return,
                benchmark_return=benchmark_return,
                excess_return=excess_return,
                portfolio_volatility=portfolio_vol,
                benchmark_volatility=benchmark_vol,
                tracking_error=tracking_error,
                alpha=alpha,
                beta=beta,
                correlation=correlation,
                information_ratio=information_ratio,
                treynor_ratio=treynor_ratio,
                selection_return=selection_return,
                timing_return=timing_return,
                max_portfolio_drawdown=portfolio_dd,
                max_benchmark_drawdown=benchmark_dd,
                relative_drawdown=relative_dd,
                alpha_p_value=alpha_pval,
                beta_p_value=beta_pval,
                calculation_date=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error comparing to benchmark {benchmark_symbol}: {e}")
            return self._empty_comparison(benchmark_symbol)

    def compare_to_multiple_benchmarks(
        self,
        portfolio_returns: Union[pd.Series, List[float]],
        benchmarks: Optional[List[str]] = None,
        portfolio_timestamps: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, BenchmarkComparison]:
        """
        Compare portfolio to multiple benchmarks in parallel.

        Args:
            portfolio_returns: Portfolio returns
            benchmarks: List of benchmark symbols
            portfolio_timestamps: Timestamps for portfolio returns

        Returns:
            Dictionary of benchmark comparisons
        """
        if benchmarks is None:
            benchmarks = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'BTC-USD']

        comparisons = {}

        # Use ThreadPoolExecutor for parallel benchmark fetching
        with ThreadPoolExecutor(max_workers=min(len(benchmarks), 6)) as executor:
            future_to_symbol = {
                executor.submit(
                    self.compare_to_benchmark,
                    portfolio_returns,
                    symbol,
                    portfolio_timestamps
                ): symbol
                for symbol in benchmarks
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    comparison = future.result(timeout=30)
                    comparisons[symbol] = comparison
                except Exception as e:
                    logger.error(f"Error processing benchmark {symbol}: {e}")
                    comparisons[symbol] = self._empty_comparison(symbol)

        return comparisons

    def get_multi_benchmark_summary(
        self,
        comparisons: Dict[str, BenchmarkComparison]
    ) -> MultiBenchmarkSummary:
        """Create summary of multi-benchmark analysis"""
        if not comparisons:
            return MultiBenchmarkSummary(
                best_alpha_benchmark="",
                best_alpha_value=0.0,
                lowest_correlation_benchmark="",
                lowest_correlation_value=1.0,
                best_information_ratio_benchmark="",
                best_information_ratio_value=0.0,
                average_alpha=0.0,
                average_beta=0.0,
                average_correlation=0.0,
                calculation_date=datetime.now()
            )

        # Find best metrics
        best_alpha = max(comparisons.items(), key=lambda x: x[1].alpha)
        lowest_corr = min(comparisons.items(), key=lambda x: abs(x[1].correlation))
        best_info_ratio = max(comparisons.items(), key=lambda x: x[1].information_ratio)

        # Calculate averages
        alphas = [c.alpha for c in comparisons.values()]
        betas = [c.beta for c in comparisons.values()]
        correlations = [c.correlation for c in comparisons.values()]

        return MultiBenchmarkSummary(
            best_alpha_benchmark=best_alpha[0],
            best_alpha_value=best_alpha[1].alpha,
            lowest_correlation_benchmark=lowest_corr[0],
            lowest_correlation_value=lowest_corr[1].correlation,
            best_information_ratio_benchmark=best_info_ratio[0],
            best_information_ratio_value=best_info_ratio[1].information_ratio,
            average_alpha=np.mean(alphas),
            average_beta=np.mean(betas),
            average_correlation=np.mean(correlations),
            calculation_date=datetime.now()
        )

    def calculate_rolling_alpha_beta(
        self,
        portfolio_returns: Union[pd.Series, List[float]],
        benchmark_symbol: str = 'SPY',
        window_days: int = 60,
        portfolio_timestamps: Optional[pd.DatetimeIndex] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling alpha and beta over time.

        Args:
            portfolio_returns: Portfolio returns
            benchmark_symbol: Benchmark symbol
            window_days: Rolling window size
            portfolio_timestamps: Timestamps for returns

        Returns:
            DataFrame with rolling alpha and beta
        """
        try:
            # Prepare data
            if isinstance(portfolio_returns, list):
                if portfolio_timestamps is None:
                    portfolio_timestamps = pd.date_range(
                        end=datetime.now(), periods=len(portfolio_returns), freq='D'
                    )
                portfolio_returns = pd.Series(portfolio_returns, index=portfolio_timestamps)

            # Get benchmark data
            benchmark_data = self._get_benchmark_data(
                benchmark_symbol,
                start_date=portfolio_returns.index[0],
                end_date=portfolio_returns.index[-1]
            )

            # Align data
            aligned_port, aligned_bench = self._align_returns(portfolio_returns, benchmark_data)

            if len(aligned_port) < window_days + 10:
                return pd.DataFrame()

            # Calculate rolling metrics
            results = []
            for i in range(window_days, len(aligned_port)):
                window_port = aligned_port.iloc[i-window_days:i]
                window_bench = aligned_bench.iloc[i-window_days:i]

                alpha, beta, alpha_pval, beta_pval = self._calculate_alpha_beta_stats(
                    window_port, window_bench
                )

                correlation = np.corrcoef(window_port, window_bench)[0, 1]
                tracking_error = (window_port - window_bench).std() * np.sqrt(252)

                results.append({
                    'date': aligned_port.index[i],
                    'alpha': alpha,
                    'beta': beta,
                    'alpha_p_value': alpha_pval,
                    'beta_p_value': beta_pval,
                    'correlation': correlation,
                    'tracking_error': tracking_error,
                    'excess_return': (window_port - window_bench).sum()
                })

            return pd.DataFrame(results)

        except Exception as e:
            logger.error(f"Error calculating rolling alpha/beta: {e}")
            return pd.DataFrame()

    def _get_benchmark_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.Series:
        """Get benchmark data with caching"""
        with self._cache_lock:
            cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}"

            # Check cache
            if cache_key in self.benchmark_cache:
                cache_entry = self.benchmark_cache[cache_key]
                cache_time = cache_entry['timestamp']
                if (datetime.now() - cache_time).total_seconds() < self.cache_duration_hours * 3600:
                    return cache_entry['data']

            try:
                # Fetch fresh data
                ticker = yf.Ticker(symbol)
                hist = ticker.history(
                    start=start_date - timedelta(days=5),  # Buffer for weekends
                    end=end_date + timedelta(days=1)
                )

                if hist.empty:
                    logger.warning(f"No data found for benchmark {symbol}")
                    return pd.Series()

                returns = hist['Close'].pct_change().dropna()

                # Cache the data
                self.benchmark_cache[cache_key] = {
                    'data': returns,
                    'timestamp': datetime.now()
                }

                # Clean old cache entries
                self._clean_cache()

                return returns

            except Exception as e:
                logger.error(f"Error fetching benchmark data for {symbol}: {e}")
                return pd.Series()

    def _align_returns(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """Align portfolio and benchmark returns by date"""
        # Find common dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)

        if len(common_dates) == 0:
            # If no exact matches, try to interpolate or use closest dates
            # For now, return empty series
            return pd.Series(), pd.Series()

        aligned_portfolio = portfolio_returns.loc[common_dates]
        aligned_benchmark = benchmark_returns.loc[common_dates]

        return aligned_portfolio, aligned_benchmark

    def _calculate_alpha_beta_stats(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Tuple[float, float, float, float]:
        """Calculate alpha, beta with statistical significance"""
        try:
            from scipy import stats

            # Prepare data
            y = portfolio_returns.values
            x = benchmark_returns.values

            if len(y) != len(x) or len(y) < 3:
                return 0.0, 1.0, 1.0, 1.0

            # Add constant for intercept (alpha)
            X = np.column_stack([np.ones(len(x)), x])

            # Perform regression
            coefficients, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

            alpha = coefficients[0]
            beta = coefficients[1]

            # Calculate standard errors and t-statistics
            if len(residuals) > 0:
                mse = residuals[0] / (len(y) - 2)
                var_coeff = mse * np.linalg.inv(X.T @ X).diagonal()
                std_errors = np.sqrt(var_coeff)

                # t-statistics and p-values
                t_alpha = alpha / std_errors[0] if std_errors[0] > 0 else 0
                t_beta = (beta - 1) / std_errors[1] if std_errors[1] > 0 else 0

                # Two-tailed p-values
                df = len(y) - 2
                alpha_p_value = 2 * (1 - stats.t.cdf(abs(t_alpha), df))
                beta_p_value = 2 * (1 - stats.t.cdf(abs(t_beta), df))
            else:
                alpha_p_value = 1.0
                beta_p_value = 1.0

            # Annualize alpha
            alpha_annualized = alpha * 252

            return alpha_annualized, beta, alpha_p_value, beta_p_value

        except Exception as e:
            logger.debug(f"Error calculating alpha/beta statistics: {e}")
            return 0.0, 1.0, 1.0, 1.0

    def _calculate_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """Calculate performance attribution (simplified version)"""
        try:
            # This is a simplified attribution analysis
            # Selection return: excess return due to security selection
            # Timing return: excess return due to market timing

            excess_returns = portfolio_returns - benchmark_returns

            # Selection return (alpha component)
            selection_return = excess_returns.mean() * 252

            # Timing return (beta timing component) - simplified
            portfolio_vol = portfolio_returns.std() * np.sqrt(252)
            benchmark_vol = benchmark_returns.std() * np.sqrt(252)
            timing_return = (portfolio_vol - benchmark_vol) * benchmark_returns.mean() * 252

            return selection_return, timing_return

        except Exception as e:
            logger.debug(f"Error calculating attribution: {e}")
            return 0.0, 0.0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except Exception as e:
            logger.debug(f"Error calculating max drawdown: {e}")
            return 0.0

    def _clean_cache(self):
        """Clean old cache entries"""
        cutoff_time = datetime.now() - timedelta(hours=self.cache_duration_hours * 2)
        keys_to_remove = [
            key for key, value in self.benchmark_cache.items()
            if value['timestamp'] < cutoff_time
        ]
        for key in keys_to_remove:
            del self.benchmark_cache[key]

    def _empty_comparison(self, benchmark_symbol: str) -> BenchmarkComparison:
        """Return empty comparison object"""
        return BenchmarkComparison(
            benchmark_symbol=benchmark_symbol,
            benchmark_name=self.benchmark_configs.get(benchmark_symbol, {}).get('name', benchmark_symbol),
            period_days=0,
            portfolio_return=0.0,
            benchmark_return=0.0,
            excess_return=0.0,
            portfolio_volatility=0.0,
            benchmark_volatility=0.0,
            tracking_error=0.0,
            alpha=0.0,
            beta=1.0,
            correlation=0.0,
            information_ratio=0.0,
            treynor_ratio=0.0,
            selection_return=0.0,
            timing_return=0.0,
            max_portfolio_drawdown=0.0,
            max_benchmark_drawdown=0.0,
            relative_drawdown=0.0,
            alpha_p_value=1.0,
            beta_p_value=1.0,
            calculation_date=datetime.now()
        )

    def get_benchmark_correlation_matrix(
        self,
        benchmarks: List[str],
        days: int = 252
    ) -> pd.DataFrame:
        """Get correlation matrix between benchmarks"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 30)

            benchmark_returns = {}

            for symbol in benchmarks:
                returns = self._get_benchmark_data(symbol, start_date, end_date)
                if not returns.empty:
                    benchmark_returns[symbol] = returns

            if len(benchmark_returns) < 2:
                return pd.DataFrame()

            # Align all returns
            common_dates = None
            for returns in benchmark_returns.values():
                if common_dates is None:
                    common_dates = returns.index
                else:
                    common_dates = common_dates.intersection(returns.index)

            aligned_returns = {}
            for symbol, returns in benchmark_returns.items():
                aligned_returns[symbol] = returns.loc[common_dates].tail(days)

            # Create correlation matrix
            returns_df = pd.DataFrame(aligned_returns)
            correlation_matrix = returns_df.corr()

            return correlation_matrix

        except Exception as e:
            logger.error(f"Error calculating benchmark correlation matrix: {e}")
            return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    # Create sample portfolio returns
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    portfolio_returns = pd.Series(
        np.random.normal(0.001, 0.02, len(dates)),
        index=dates
    )

    # Initialize analyzer
    analyzer = BenchmarkAnalyzer()

    # Single benchmark comparison
    spy_comparison = analyzer.compare_to_benchmark(portfolio_returns, 'SPY')
    print(f"SPY Comparison:")
    print(f"Alpha: {spy_comparison.alpha:.3f}")
    print(f"Beta: {spy_comparison.beta:.3f}")
    print(f"Information Ratio: {spy_comparison.information_ratio:.3f}")

    # Multiple benchmark comparison
    benchmarks = ['SPY', 'QQQ', 'TLT', 'GLD']
    multi_comparison = analyzer.compare_to_multiple_benchmarks(portfolio_returns, benchmarks)

    print(f"\nMultiple Benchmark Comparison:")
    for symbol, comparison in multi_comparison.items():
        print(f"{symbol}: Alpha={comparison.alpha:.3f}, Beta={comparison.beta:.3f}")

    # Multi-benchmark summary
    summary = analyzer.get_multi_benchmark_summary(multi_comparison)
    print(f"\nBest Alpha Benchmark: {summary.best_alpha_benchmark} ({summary.best_alpha_value:.3f})")
    print(f"Average Beta: {summary.average_beta:.3f}")

    # Rolling analysis
    rolling_ab = analyzer.calculate_rolling_alpha_beta(portfolio_returns, 'SPY', window_days=60)
    if not rolling_ab.empty:
        print(f"\nRolling Analysis (last 5 observations):")
        print(rolling_ab.tail())

    # Correlation matrix
    corr_matrix = analyzer.get_benchmark_correlation_matrix(['SPY', 'QQQ', 'TLT', 'GLD'])
    if not corr_matrix.empty:
        print(f"\nBenchmark Correlation Matrix:")
        print(corr_matrix)