"""
Performance Metrics Analyzer
===========================

Comprehensive performance analysis for ML trading models.
Provides detailed metrics, visualizations, and model monitoring capabilities.

Author: Trading Bot Arsenal
Created: February 2026
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    model_name: str
    period_start: datetime
    period_end: datetime

    # Return metrics
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Risk metrics
    max_drawdown: float
    var_95: float  # Value at Risk
    cvar_95: float  # Conditional VaR
    beta: float
    alpha: float

    # Trading metrics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int

    # Model-specific metrics
    prediction_accuracy: float
    precision: float
    recall: float
    f1_score: float

    # Additional analysis
    monthly_returns: pd.Series
    drawdown_periods: List[Dict]
    correlation_with_benchmark: float


class PerformanceAnalyzer:
    """
    Advanced performance analysis for ML trading models.

    Features:
    - Comprehensive risk-return metrics
    - Drawdown analysis
    - Model prediction accuracy tracking
    - Benchmark comparison
    - Risk attribution analysis
    - Performance visualization data
    """

    def __init__(self, benchmark_symbol: str = "SPY", risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer.

        Args:
            benchmark_symbol: Benchmark for comparison
            risk_free_rate: Annual risk-free rate
        """
        self.benchmark_symbol = benchmark_symbol
        self.risk_free_rate = risk_free_rate

    def analyze_performance(self,
                          returns: pd.Series,
                          predictions: pd.Series = None,
                          actual_outcomes: pd.Series = None,
                          benchmark_returns: pd.Series = None,
                          model_name: str = "ML Model") -> PerformanceReport:
        """
        Comprehensive performance analysis.

        Args:
            returns: Portfolio returns time series
            predictions: Model predictions
            actual_outcomes: Actual outcomes for prediction accuracy
            benchmark_returns: Benchmark returns for comparison
            model_name: Name of the model

        Returns:
            Comprehensive performance report
        """
        logger.info(f"Analyzing performance for {model_name}")

        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        annual_return = self._annualized_return(returns)
        volatility = returns.std() * np.sqrt(252)

        # Risk-adjusted returns
        excess_returns = returns - self.risk_free_rate / 252
        sharpe_ratio = self._sharpe_ratio(returns)
        sortino_ratio = self._sortino_ratio(returns)
        calmar_ratio = self._calmar_ratio(returns)

        # Drawdown analysis
        max_drawdown, drawdown_periods = self._drawdown_analysis(returns)

        # Risk metrics
        var_95 = self._value_at_risk(returns, confidence=0.95)
        cvar_95 = self._conditional_var(returns, confidence=0.95)

        # Benchmark comparison
        beta, alpha = self._benchmark_analysis(returns, benchmark_returns)
        correlation_with_benchmark = self._benchmark_correlation(returns, benchmark_returns)

        # Trading metrics (if available)
        win_rate, profit_factor, avg_win, avg_loss, total_trades = self._trading_metrics(returns)

        # Model prediction metrics
        prediction_accuracy, precision, recall, f1_score = self._prediction_metrics(
            predictions, actual_outcomes
        )

        # Monthly analysis
        monthly_returns = self._monthly_returns_analysis(returns)

        return PerformanceReport(
            model_name=model_name,
            period_start=returns.index[0],
            period_end=returns.index[-1],
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            alpha=alpha,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=total_trades,
            prediction_accuracy=prediction_accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            monthly_returns=monthly_returns,
            drawdown_periods=drawdown_periods,
            correlation_with_benchmark=correlation_with_benchmark
        )

    def _annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        if len(returns) == 0:
            return 0.0

        total_return = (1 + returns).prod() - 1
        years = len(returns) / 252  # Approximate trading days per year

        if years <= 0:
            return 0.0

        return (1 + total_return) ** (1 / years) - 1

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0

        returns_std = float(returns.std())
        if returns_std == 0:
            return 0.0

        excess_returns = returns - self.risk_free_rate / 252
        return float(excess_returns.mean()) / returns_std * np.sqrt(252)

    def _sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = returns[returns < 0]
        excess_mean = float(excess_returns.mean())

        if len(downside_returns) == 0:
            return np.inf if excess_mean > 0 else 0.0

        downside_std = float(downside_returns.std())
        if downside_std == 0:
            return np.inf if excess_mean > 0 else 0.0

        return excess_mean / downside_std * np.sqrt(252)

    def _calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        annual_return = float(self._annualized_return(returns))
        max_drawdown, _ = self._drawdown_analysis(returns)
        max_drawdown = float(max_drawdown)

        if max_drawdown == 0:
            return np.inf if annual_return > 0 else 0.0

        return annual_return / abs(max_drawdown)

    def _drawdown_analysis(self, returns: pd.Series) -> Tuple[float, List[Dict]]:
        """Analyze drawdowns"""
        if len(returns) == 0:
            return 0.0, []

        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()

        # Calculate running maximum
        running_max = cum_returns.cummax()

        # Calculate drawdown
        drawdown = (cum_returns - running_max) / running_max

        # Find maximum drawdown
        max_drawdown = drawdown.min()

        # Identify drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_date = None

        for date, dd in drawdown.items():
            if dd < -0.01 and not in_drawdown:  # Start of drawdown (>1%)
                in_drawdown = True
                start_date = date
            elif dd >= -0.01 and in_drawdown:  # End of drawdown
                in_drawdown = False
                if start_date:
                    period_dd = drawdown[start_date:date]
                    drawdown_periods.append({
                        'start': start_date,
                        'end': date,
                        'duration': (date - start_date).days,
                        'max_drawdown': period_dd.min()
                    })

        return max_drawdown, drawdown_periods

    def _value_at_risk(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0

        return np.percentile(returns, (1 - confidence) * 100)

    def _conditional_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0

        var = self._value_at_risk(returns, confidence)
        return returns[returns <= var].mean()

    def _benchmark_analysis(self, returns: pd.Series,
                          benchmark_returns: pd.Series = None) -> Tuple[float, float]:
        """Calculate beta and alpha vs benchmark"""
        if benchmark_returns is None or len(returns) != len(benchmark_returns):
            return 1.0, 0.0  # Default values

        # Align returns
        aligned_data = pd.concat([returns, benchmark_returns], axis=1, join='inner')
        if aligned_data.empty:
            return 1.0, 0.0

        portfolio_returns = aligned_data.iloc[:, 0]
        bench_returns = aligned_data.iloc[:, 1]

        # Calculate beta
        covariance = float(portfolio_returns.cov(bench_returns))
        benchmark_variance = float(bench_returns.var())

        if benchmark_variance == 0:
            beta = 1.0
        else:
            beta = covariance / benchmark_variance

        # Calculate alpha
        portfolio_annual = self._annualized_return(portfolio_returns)
        benchmark_annual = self._annualized_return(bench_returns)

        alpha = portfolio_annual - (self.risk_free_rate + beta * (benchmark_annual - self.risk_free_rate))

        return beta, alpha

    def _benchmark_correlation(self, returns: pd.Series,
                             benchmark_returns: pd.Series = None) -> float:
        """Calculate correlation with benchmark"""
        if benchmark_returns is None:
            return 0.0

        aligned_data = pd.concat([returns, benchmark_returns], axis=1, join='inner')
        if aligned_data.empty or len(aligned_data) < 2:
            return 0.0

        return aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])

    def _trading_metrics(self, returns: pd.Series) -> Tuple[float, float, float, float, int]:
        """Calculate trading-specific metrics"""
        if len(returns) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0

        # Identify individual trades (non-zero returns)
        trades = returns[returns != 0]

        if len(trades) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0

        # Win/loss analysis
        winning_trades = trades[trades > 0]
        losing_trades = trades[trades < 0]

        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0.0
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0.0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0.0

        # Profit factor
        total_wins = winning_trades.sum() if len(winning_trades) > 0 else 0.0
        total_losses = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0.0

        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf

        return float(win_rate), float(profit_factor), float(avg_win), float(avg_loss), len(trades)

    def _prediction_metrics(self, predictions: pd.Series = None,
                          actual_outcomes: pd.Series = None) -> Tuple[float, float, float, float]:
        """Calculate model prediction metrics"""
        if predictions is None or actual_outcomes is None:
            return 0.0, 0.0, 0.0, 0.0

        # Align predictions and outcomes
        aligned_data = pd.concat([predictions, actual_outcomes], axis=1, join='inner')
        if aligned_data.empty:
            return 0.0, 0.0, 0.0, 0.0

        pred = aligned_data.iloc[:, 0]
        actual = aligned_data.iloc[:, 1]

        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            # Convert to binary if needed
            if pred.nunique() > 2:
                pred_binary = (pred > 0).astype(int)
                actual_binary = (actual > 0).astype(int)
            else:
                pred_binary = pred
                actual_binary = actual

            accuracy = accuracy_score(actual_binary, pred_binary)
            precision = precision_score(actual_binary, pred_binary, average='weighted', zero_division=0)
            recall = recall_score(actual_binary, pred_binary, average='weighted', zero_division=0)
            f1 = f1_score(actual_binary, pred_binary, average='weighted', zero_division=0)

            return accuracy, precision, recall, f1

        except ImportError:
            # Fallback calculation
            accuracy = float((pred == actual).mean()) if len(pred) > 0 else 0.0
            return accuracy, 0.0, 0.0, 0.0
        except Exception as e:
            logger.debug(f"Prediction metrics calculation failed: {e}")
            return 0.0, 0.0, 0.0, 0.0

    def _monthly_returns_analysis(self, returns: pd.Series) -> pd.Series:
        """Analyze monthly returns"""
        if len(returns) == 0:
            return pd.Series()

        try:
            # Group by month and calculate monthly returns
            monthly_data = returns.groupby(pd.Grouper(freq='M')).apply(
                lambda x: (1 + x).prod() - 1
            )
            return monthly_data
        except Exception as e:
            logger.debug(f"Monthly analysis failed: {e}")
            return pd.Series()

    def compare_models(self, reports: List[PerformanceReport]) -> Dict[str, Any]:
        """
        Compare performance across multiple models.

        Args:
            reports: List of performance reports to compare

        Returns:
            Comparison analysis
        """
        if not reports:
            return {}

        comparison = {
            'models': [r.model_name for r in reports],
            'metrics_comparison': {}
        }

        # Compare key metrics
        metrics_to_compare = [
            'total_return', 'annual_return', 'volatility', 'sharpe_ratio',
            'max_drawdown', 'win_rate', 'prediction_accuracy'
        ]

        for metric in metrics_to_compare:
            values = [getattr(report, metric, 0) for report in reports]
            comparison['metrics_comparison'][metric] = {
                'values': values,
                'best_model': reports[np.argmax(values) if metric != 'max_drawdown' else np.argmin(np.abs(values))].model_name,
                'best_value': max(values) if metric != 'max_drawdown' else min(values, key=abs)
            }

        # Risk-adjusted ranking
        risk_adj_scores = []
        for report in reports:
            # Simple risk-adjusted score (Sharpe ratio weighted)
            score = report.sharpe_ratio - abs(report.max_drawdown) * 0.5
            risk_adj_scores.append(score)

        best_risk_adj_idx = np.argmax(risk_adj_scores)
        comparison['best_risk_adjusted'] = {
            'model': reports[best_risk_adj_idx].model_name,
            'score': risk_adj_scores[best_risk_adj_idx]
        }

        return comparison

    def generate_summary_statistics(self, report: PerformanceReport) -> Dict[str, Any]:
        """Generate summary statistics for dashboard display"""
        return {
            'model_name': report.model_name,
            'period': {
                'start': report.period_start.strftime('%Y-%m-%d'),
                'end': report.period_end.strftime('%Y-%m-%d'),
                'days': (report.period_end - report.period_start).days
            },
            'returns': {
                'total': f"{report.total_return:.2%}",
                'annual': f"{report.annual_return:.2%}",
                'volatility': f"{report.volatility:.2%}",
                'sharpe': f"{report.sharpe_ratio:.2f}"
            },
            'risk': {
                'max_drawdown': f"{report.max_drawdown:.2%}",
                'var_95': f"{report.var_95:.2%}",
                'beta': f"{report.beta:.2f}"
            },
            'trading': {
                'win_rate': f"{report.win_rate:.1%}",
                'total_trades': report.total_trades,
                'profit_factor': f"{report.profit_factor:.2f}"
            },
            'model_performance': {
                'accuracy': f"{report.prediction_accuracy:.1%}",
                'precision': f"{report.precision:.1%}",
                'f1_score': f"{report.f1_score:.2f}"
            }
        }


def main():
    """Test performance analyzer"""
    print("Testing Performance Analyzer")
    print("=" * 35)

    # Generate sample returns data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

    # Generate realistic returns with some trend and volatility
    base_returns = np.random.normal(0.0008, 0.015, len(dates))  # ~20% annual volatility
    trend = np.linspace(0, 0.0005, len(dates))  # Small positive trend
    returns = pd.Series(base_returns + trend, index=dates)

    # Generate sample predictions
    predictions = pd.Series(np.random.choice([-1, 0, 1], len(dates)), index=dates)
    actual_outcomes = pd.Series(np.random.choice([-1, 0, 1], len(dates)), index=dates)

    # Initialize analyzer
    analyzer = PerformanceAnalyzer()

    # Analyze performance
    print("Analyzing performance...")
    report = analyzer.analyze_performance(
        returns=returns,
        predictions=predictions,
        actual_outcomes=actual_outcomes,
        model_name="Test ML Model"
    )

    # Display results
    print(f"\nPerformance Report for {report.model_name}")
    print(f"Period: {report.period_start.date()} to {report.period_end.date()}")
    print(f"Total Return: {report.total_return:.2%}")
    print(f"Annual Return: {report.annual_return:.2%}")
    print(f"Volatility: {report.volatility:.2%}")
    print(f"Sharpe Ratio: {report.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {report.max_drawdown:.2%}")
    print(f"Win Rate: {report.win_rate:.1%}")
    print(f"Prediction Accuracy: {report.prediction_accuracy:.1%}")

    # Generate summary statistics
    print("\nSummary Statistics:")
    summary = analyzer.generate_summary_statistics(report)
    for category, metrics in summary.items():
        if isinstance(metrics, dict):
            print(f"\n{category.upper()}:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        else:
            print(f"{category}: {metrics}")

    print("\nPerformance analyzer test completed!")


if __name__ == "__main__":
    main()