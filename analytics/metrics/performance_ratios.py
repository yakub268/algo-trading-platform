"""
Performance Ratios Calculator
============================

Advanced performance ratio calculations including:
- Sharpe Ratio (risk-adjusted returns)
- Sortino Ratio (downside deviation focus)
- Calmar Ratio (return vs max drawdown)
- Information Ratio (excess return vs tracking error)
- Treynor Ratio (return vs beta)
- Jensen's Alpha (risk-adjusted excess return)
- Omega Ratio (gains vs losses)

Author: Trading Bot System
Created: February 2026
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import yfinance as yf
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RatioResults:
    """Container for all calculated performance ratios"""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: float
    jensen_alpha: float
    omega_ratio: float
    beta: float
    correlation: float
    tracking_error: float
    calculation_date: datetime
    period_days: int

    def to_dict(self) -> Dict[str, Union[float, str, int]]:
        """Convert to dictionary"""
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'information_ratio': self.information_ratio,
            'treynor_ratio': self.treynor_ratio,
            'jensen_alpha': self.jensen_alpha,
            'omega_ratio': self.omega_ratio,
            'beta': self.beta,
            'correlation': self.correlation,
            'tracking_error': self.tracking_error,
            'calculation_date': self.calculation_date.isoformat(),
            'period_days': self.period_days
        }


class PerformanceRatios:
    """
    Comprehensive performance ratio calculator for trading strategies.

    Calculates various risk-adjusted performance metrics using portfolio
    returns and benchmark comparisons.

    Features:
    - Multiple ratio calculations with industry standard formulas
    - Benchmark comparison capabilities
    - Rolling ratio calculations
    - Confidence intervals for ratios
    - Risk-free rate integration
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance ratios calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.benchmark_cache: Dict[str, pd.DataFrame] = {}

        logger.info(f"Performance Ratios initialized (risk-free rate: {risk_free_rate:.2%})")

    def calculate_all_ratios(
        self,
        returns: Union[pd.Series, List[float], np.ndarray],
        benchmark_returns: Optional[Union[pd.Series, List[float], np.ndarray]] = None,
        benchmark_symbol: str = 'SPY',
        period_days: Optional[int] = None
    ) -> RatioResults:
        """
        Calculate all performance ratios.

        Args:
            returns: Portfolio returns (daily)
            benchmark_returns: Benchmark returns (if None, will fetch)
            benchmark_symbol: Benchmark symbol for fetching
            period_days: Period for calculation

        Returns:
            RatioResults object with all calculated ratios
        """
        # Convert returns to pandas Series
        if isinstance(returns, (list, np.ndarray)):
            returns = pd.Series(returns)

        if len(returns) == 0:
            return self._empty_results()

        # Get benchmark returns if not provided
        if benchmark_returns is None:
            benchmark_returns = self._get_benchmark_returns(
                benchmark_symbol,
                len(returns)
            )
        elif isinstance(benchmark_returns, (list, np.ndarray)):
            benchmark_returns = pd.Series(benchmark_returns)

        # Align returns if benchmark available
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            min_length = min(len(returns), len(benchmark_returns))
            returns = returns.tail(min_length)
            benchmark_returns = benchmark_returns.tail(min_length)

        # Calculate daily risk-free rate
        daily_rf_rate = self.risk_free_rate / 252

        # Calculate individual ratios
        sharpe = self._calculate_sharpe_ratio(returns, daily_rf_rate)
        sortino = self._calculate_sortino_ratio(returns, daily_rf_rate)
        calmar = self._calculate_calmar_ratio(returns)
        omega = self._calculate_omega_ratio(returns)

        # Benchmark-dependent ratios
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            info_ratio = self._calculate_information_ratio(returns, benchmark_returns)
            treynor = self._calculate_treynor_ratio(returns, benchmark_returns, daily_rf_rate)
            jensen_alpha = self._calculate_jensen_alpha(returns, benchmark_returns, daily_rf_rate)
            beta = self._calculate_beta(returns, benchmark_returns)
            correlation = self._calculate_correlation(returns, benchmark_returns)
            tracking_error = self._calculate_tracking_error(returns, benchmark_returns)
        else:
            info_ratio = 0.0
            treynor = 0.0
            jensen_alpha = 0.0
            beta = 1.0
            correlation = 0.0
            tracking_error = 0.0

        return RatioResults(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=info_ratio,
            treynor_ratio=treynor,
            jensen_alpha=jensen_alpha,
            omega_ratio=omega,
            beta=beta,
            correlation=correlation,
            tracking_error=tracking_error,
            calculation_date=datetime.now(),
            period_days=period_days or len(returns)
        )

    def _calculate_sharpe_ratio(self, returns: pd.Series, daily_rf_rate: float) -> float:
        """
        Calculate Sharpe Ratio.

        Sharpe Ratio = (Portfolio Return - Risk Free Rate) / Portfolio Volatility
        """
        try:
            if len(returns) == 0:
                return 0.0

            excess_returns = returns - daily_rf_rate

            if excess_returns.std() == 0:
                return 0.0

            # Annualize
            sharpe = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))

            return float(sharpe) if not np.isnan(sharpe) and not np.isinf(sharpe) else 0.0

        except Exception as e:
            logger.debug(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def _calculate_sortino_ratio(self, returns: pd.Series, daily_rf_rate: float) -> float:
        """
        Calculate Sortino Ratio.

        Sortino Ratio = (Portfolio Return - Risk Free Rate) / Downside Deviation
        """
        try:
            if len(returns) == 0:
                return 0.0

            excess_returns = returns - daily_rf_rate
            downside_returns = excess_returns[excess_returns < 0]

            if len(downside_returns) == 0 or downside_returns.std() == 0:
                return 0.0

            # Annualize
            sortino = (excess_returns.mean() * 252) / (downside_returns.std() * np.sqrt(252))

            return float(sortino) if not np.isnan(sortino) and not np.isinf(sortino) else 0.0

        except Exception as e:
            logger.debug(f"Error calculating Sortino ratio: {e}")
            return 0.0

    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Calmar Ratio.

        Calmar Ratio = Annualized Return / Maximum Drawdown
        """
        try:
            if len(returns) == 0:
                return 0.0

            # Calculate cumulative returns
            cumulative = (1 + returns).cumprod()

            # Calculate running maximum
            running_max = cumulative.expanding().max()

            # Calculate drawdowns
            drawdowns = (cumulative - running_max) / running_max

            max_drawdown = drawdowns.min()

            if max_drawdown == 0:
                return 0.0

            annualized_return = returns.mean() * 252

            calmar = annualized_return / abs(max_drawdown)

            return float(calmar) if not np.isnan(calmar) and not np.isinf(calmar) else 0.0

        except Exception as e:
            logger.debug(f"Error calculating Calmar ratio: {e}")
            return 0.0

    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """
        Calculate Omega Ratio.

        Omega Ratio = Sum of gains above threshold / Sum of losses below threshold
        """
        try:
            if len(returns) == 0:
                return 0.0

            gains = returns[returns > threshold] - threshold
            losses = threshold - returns[returns < threshold]

            if losses.sum() == 0:
                return float('inf') if gains.sum() > 0 else 0.0

            omega = gains.sum() / losses.sum()

            return float(omega) if not np.isnan(omega) and not np.isinf(omega) else 0.0

        except Exception as e:
            logger.debug(f"Error calculating Omega ratio: {e}")
            return 0.0

    def _calculate_information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate Information Ratio.

        Information Ratio = Excess Return / Tracking Error
        """
        try:
            if len(returns) == 0 or len(benchmark_returns) == 0:
                return 0.0

            excess_returns = returns - benchmark_returns
            tracking_error = excess_returns.std()

            if tracking_error == 0:
                return 0.0

            # Annualize
            info_ratio = (excess_returns.mean() * 252) / (tracking_error * np.sqrt(252))

            return float(info_ratio) if not np.isnan(info_ratio) and not np.isinf(info_ratio) else 0.0

        except Exception as e:
            logger.debug(f"Error calculating Information ratio: {e}")
            return 0.0

    def _calculate_treynor_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        daily_rf_rate: float
    ) -> float:
        """
        Calculate Treynor Ratio.

        Treynor Ratio = (Portfolio Return - Risk Free Rate) / Beta
        """
        try:
            if len(returns) == 0 or len(benchmark_returns) == 0:
                return 0.0

            beta = self._calculate_beta(returns, benchmark_returns)

            if beta == 0:
                return 0.0

            excess_returns = returns - daily_rf_rate

            # Annualize
            treynor = (excess_returns.mean() * 252) / beta

            return float(treynor) if not np.isnan(treynor) and not np.isinf(treynor) else 0.0

        except Exception as e:
            logger.debug(f"Error calculating Treynor ratio: {e}")
            return 0.0

    def _calculate_jensen_alpha(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        daily_rf_rate: float
    ) -> float:
        """
        Calculate Jensen's Alpha.

        Alpha = Portfolio Return - [Risk Free Rate + Beta * (Benchmark Return - Risk Free Rate)]
        """
        try:
            if len(returns) == 0 or len(benchmark_returns) == 0:
                return 0.0

            beta = self._calculate_beta(returns, benchmark_returns)

            portfolio_excess = returns - daily_rf_rate
            benchmark_excess = benchmark_returns - daily_rf_rate

            expected_excess = beta * benchmark_excess
            alpha_series = portfolio_excess - expected_excess

            # Annualize
            alpha = alpha_series.mean() * 252

            return float(alpha) if not np.isnan(alpha) and not np.isinf(alpha) else 0.0

        except Exception as e:
            logger.debug(f"Error calculating Jensen's Alpha: {e}")
            return 0.0

    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate Beta (sensitivity to benchmark).

        Beta = Covariance(Portfolio, Benchmark) / Variance(Benchmark)
        """
        try:
            if len(returns) == 0 or len(benchmark_returns) == 0:
                return 1.0

            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)

            if benchmark_variance == 0:
                return 1.0

            beta = covariance / benchmark_variance

            return float(beta) if not np.isnan(beta) and not np.isinf(beta) else 1.0

        except Exception as e:
            logger.debug(f"Error calculating Beta: {e}")
            return 1.0

    def _calculate_correlation(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate correlation with benchmark"""
        try:
            if len(returns) == 0 or len(benchmark_returns) == 0:
                return 0.0

            correlation = np.corrcoef(returns, benchmark_returns)[0, 1]

            return float(correlation) if not np.isnan(correlation) else 0.0

        except Exception as e:
            logger.debug(f"Error calculating correlation: {e}")
            return 0.0

    def _calculate_tracking_error(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate tracking error (standard deviation of excess returns)"""
        try:
            if len(returns) == 0 or len(benchmark_returns) == 0:
                return 0.0

            excess_returns = returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)  # Annualized

            return float(tracking_error) if not np.isnan(tracking_error) else 0.0

        except Exception as e:
            logger.debug(f"Error calculating tracking error: {e}")
            return 0.0

    def _get_benchmark_returns(self, symbol: str, periods: int) -> Optional[pd.Series]:
        """Get benchmark returns from cache or fetch from API"""
        try:
            # Check cache first
            if symbol in self.benchmark_cache:
                cached_data = self.benchmark_cache[symbol]
                if len(cached_data) >= periods:
                    return cached_data['returns'].tail(periods)

            # Fetch new data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{periods + 10}d")  # Extra buffer

            if len(hist) > 0:
                returns = hist['Close'].pct_change().dropna()

                # Cache the data
                self.benchmark_cache[symbol] = {
                    'data': hist,
                    'returns': returns,
                    'last_update': datetime.now()
                }

                return returns.tail(periods)

            return None

        except Exception as e:
            logger.debug(f"Error fetching benchmark returns for {symbol}: {e}")
            return None

    def calculate_rolling_ratios(
        self,
        returns: Union[pd.Series, List[float]],
        window_days: int = 30,
        benchmark_symbol: str = 'SPY'
    ) -> pd.DataFrame:
        """
        Calculate rolling performance ratios.

        Args:
            returns: Portfolio returns
            window_days: Rolling window size
            benchmark_symbol: Benchmark for comparison

        Returns:
            DataFrame with rolling ratios
        """
        try:
            if isinstance(returns, list):
                returns = pd.Series(returns)

            if len(returns) < window_days:
                return pd.DataFrame()

            # Get benchmark data
            benchmark_returns = self._get_benchmark_returns(benchmark_symbol, len(returns))

            rolling_results = []

            for i in range(window_days, len(returns) + 1):
                window_returns = returns.iloc[i-window_days:i]
                window_benchmark = benchmark_returns.iloc[i-window_days:i] if benchmark_returns is not None else None

                ratios = self.calculate_all_ratios(
                    window_returns,
                    window_benchmark,
                    period_days=window_days
                )

                result_dict = ratios.to_dict()
                result_dict['date'] = returns.index[i-1] if hasattr(returns, 'index') else i-1
                rolling_results.append(result_dict)

            return pd.DataFrame(rolling_results)

        except Exception as e:
            logger.error(f"Error calculating rolling ratios: {e}")
            return pd.DataFrame()

    def get_ratio_confidence_intervals(
        self,
        returns: Union[pd.Series, List[float]],
        confidence_level: float = 0.95,
        bootstrap_samples: int = 1000
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate confidence intervals for performance ratios using bootstrap.

        Args:
            returns: Portfolio returns
            confidence_level: Confidence level (0.95 for 95%)
            bootstrap_samples: Number of bootstrap samples

        Returns:
            Dictionary with confidence intervals for each ratio
        """
        try:
            if isinstance(returns, list):
                returns = pd.Series(returns)

            if len(returns) < 10:
                return {}

            # Bootstrap sampling
            ratio_samples = {
                'sharpe_ratio': [],
                'sortino_ratio': [],
                'calmar_ratio': []
            }

            for _ in range(bootstrap_samples):
                # Sample with replacement
                sample_returns = returns.sample(n=len(returns), replace=True)

                # Calculate ratios for sample
                ratios = self.calculate_all_ratios(sample_returns)

                ratio_samples['sharpe_ratio'].append(ratios.sharpe_ratio)
                ratio_samples['sortino_ratio'].append(ratios.sortino_ratio)
                ratio_samples['calmar_ratio'].append(ratios.calmar_ratio)

            # Calculate confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            confidence_intervals = {}
            for ratio_name, samples in ratio_samples.items():
                samples = np.array(samples)
                samples = samples[~np.isnan(samples) & ~np.isinf(samples)]

                if len(samples) > 0:
                    lower_bound = np.percentile(samples, lower_percentile)
                    upper_bound = np.percentile(samples, upper_percentile)
                    confidence_intervals[ratio_name] = (float(lower_bound), float(upper_bound))
                else:
                    confidence_intervals[ratio_name] = (0.0, 0.0)

            return confidence_intervals

        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {e}")
            return {}

    def _empty_results(self) -> RatioResults:
        """Return empty results object"""
        return RatioResults(
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            information_ratio=0.0,
            treynor_ratio=0.0,
            jensen_alpha=0.0,
            omega_ratio=0.0,
            beta=1.0,
            correlation=0.0,
            tracking_error=0.0,
            calculation_date=datetime.now(),
            period_days=0
        )

    def compare_to_benchmarks(
        self,
        returns: Union[pd.Series, List[float]],
        benchmarks: List[str] = ['SPY', 'QQQ', 'IWM', 'BTC-USD', 'GLD']
    ) -> Dict[str, RatioResults]:
        """
        Compare performance ratios against multiple benchmarks.

        Args:
            returns: Portfolio returns
            benchmarks: List of benchmark symbols

        Returns:
            Dictionary with ratio results for each benchmark
        """
        results = {}

        for benchmark in benchmarks:
            try:
                ratios = self.calculate_all_ratios(returns, benchmark_symbol=benchmark)
                results[benchmark] = ratios
            except Exception as e:
                logger.debug(f"Error calculating ratios for benchmark {benchmark}: {e}")
                results[benchmark] = self._empty_results()

        return results


# Example usage
if __name__ == "__main__":
    # Create sample returns
    np.random.seed(42)
    sample_returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns for 1 year

    # Initialize calculator
    calculator = PerformanceRatios(risk_free_rate=0.02)

    # Calculate all ratios
    ratios = calculator.calculate_all_ratios(sample_returns)

    print("Performance Ratios:")
    print(f"Sharpe Ratio: {ratios.sharpe_ratio:.3f}")
    print(f"Sortino Ratio: {ratios.sortino_ratio:.3f}")
    print(f"Calmar Ratio: {ratios.calmar_ratio:.3f}")
    print(f"Information Ratio: {ratios.information_ratio:.3f}")
    print(f"Beta: {ratios.beta:.3f}")

    # Calculate confidence intervals
    confidence_intervals = calculator.get_ratio_confidence_intervals(sample_returns)
    print(f"\n95% Confidence Intervals:")
    for ratio, (lower, upper) in confidence_intervals.items():
        print(f"{ratio}: [{lower:.3f}, {upper:.3f}]")