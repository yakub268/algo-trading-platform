"""
Monte Carlo Simulation for Strategy Validation

Robust strategy validation through:
- Trade sequence shuffling (1000+ iterations)
- Random trade skipping (simulate missed executions)
- Confidence interval calculation
- Equity curve distribution visualization

Purpose: Distinguish skill from luck by testing if strategy
performs across many possible trade orderings.

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MonteCarlo')


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results"""
    iterations: int
    original_return: float
    original_sharpe: float
    
    # Percentile statistics
    return_5th: float
    return_25th: float
    return_50th: float  # Median
    return_75th: float
    return_95th: float
    
    sharpe_5th: float
    sharpe_25th: float
    sharpe_50th: float
    sharpe_75th: float
    sharpe_95th: float
    
    drawdown_5th: float
    drawdown_50th: float
    drawdown_95th: float
    
    # Probability metrics
    prob_profitable: float  # % of simulations profitable
    prob_beat_benchmark: float  # % beating buy & hold
    prob_positive_sharpe: float  # % with Sharpe > 0
    
    # Risk metrics
    expected_max_drawdown: float
    worst_case_drawdown: float  # 95th percentile
    var_95: float  # Value at Risk (5th percentile return)
    cvar_95: float  # Conditional VaR (avg of worst 5%)
    
    # Validation
    passes_validation: bool
    validation_message: str
    
    # Raw data
    all_returns: np.ndarray
    all_sharpes: np.ndarray
    all_drawdowns: np.ndarray
    equity_curves: Optional[np.ndarray] = None


class MonteCarloSimulator:
    """
    Monte Carlo Simulation for Trading Strategy Validation
    
    Methods:
    1. Trade Shuffling: Randomize order of historical trades
    2. Trade Skipping: Randomly skip 5-15% of trades
    3. Bootstrap: Sample trades with replacement
    
    Validation Criteria:
    - 95% of simulations must be profitable
    - Median Sharpe > 0.5
    - 95th percentile drawdown < 30%
    """
    
    DEFAULT_ITERATIONS = 1000
    SKIP_RATE_MIN = 0.05  # 5% minimum trades skipped
    SKIP_RATE_MAX = 0.15  # 15% maximum trades skipped
    
    # Validation thresholds
    MIN_PROFITABLE_PCT = 0.95  # 95% must be profitable
    MIN_MEDIAN_SHARPE = 0.5
    MAX_95_DRAWDOWN = 0.30  # 30%
    
    def __init__(self, seed: int = 42):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        logger.info(f"MonteCarloSimulator initialized (seed={seed})")
    
    def simulate_from_trades(
        self,
        trades: pd.DataFrame,
        initial_capital: float = 10000.0,
        iterations: int = None,
        n_iterations: int = None,  # Alias for iterations
        method: str = 'shuffle'
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation from historical trades.
        
        Args:
            trades: DataFrame with 'return' or 'pnl_pct' column
            initial_capital: Starting capital
            iterations: Number of simulations
            method: 'shuffle', 'skip', or 'bootstrap'
            
        Returns:
            MonteCarloResult with statistics
        """
        # Support both 'iterations' and 'n_iterations' parameter names
        iterations = iterations or n_iterations or self.DEFAULT_ITERATIONS

        # Extract returns
        if 'return' in trades.columns:
            returns = trades['return'].values
        elif 'pnl_pct' in trades.columns:
            returns = trades['pnl_pct'].values
        elif 'Return' in trades.columns:
            returns = trades['Return'].values
        else:
            raise ValueError("Trades DataFrame must have 'return' or 'pnl_pct' column")
        
        returns = np.array(returns, dtype=float)
        returns = returns[~np.isnan(returns)]  # Remove NaN
        
        if len(returns) < 10:
            raise ValueError(f"Need at least 10 trades, got {len(returns)}")
        
        logger.info(f"Running {iterations} Monte Carlo simulations on {len(returns)} trades...")
        
        # Run simulations
        all_final_returns = []
        all_sharpes = []
        all_max_drawdowns = []
        equity_curves = []
        
        for i in range(iterations):
            if method == 'shuffle':
                sim_returns = self._shuffle_trades(returns)
            elif method == 'skip':
                sim_returns = self._skip_trades(returns)
            elif method == 'bootstrap':
                sim_returns = self._bootstrap_trades(returns)
            else:
                sim_returns = self._shuffle_trades(returns)
            
            # Calculate equity curve
            equity = self._calculate_equity_curve(sim_returns, initial_capital)
            equity_curves.append(equity)
            
            # Calculate metrics
            final_return = (equity[-1] - initial_capital) / initial_capital
            sharpe = self._calculate_sharpe(sim_returns)
            max_dd = self._calculate_max_drawdown(equity)
            
            all_final_returns.append(final_return)
            all_sharpes.append(sharpe)
            all_max_drawdowns.append(max_dd)
        
        all_final_returns = np.array(all_final_returns)
        all_sharpes = np.array(all_sharpes)
        all_max_drawdowns = np.array(all_max_drawdowns)
        equity_curves = np.array(equity_curves)
        
        # Calculate original (non-shuffled) metrics
        original_equity = self._calculate_equity_curve(returns, initial_capital)
        original_return = (original_equity[-1] - initial_capital) / initial_capital
        original_sharpe = self._calculate_sharpe(returns)
        
        # Calculate percentiles
        result = MonteCarloResult(
            iterations=iterations,
            original_return=original_return,
            original_sharpe=original_sharpe,
            
            return_5th=np.percentile(all_final_returns, 5),
            return_25th=np.percentile(all_final_returns, 25),
            return_50th=np.percentile(all_final_returns, 50),
            return_75th=np.percentile(all_final_returns, 75),
            return_95th=np.percentile(all_final_returns, 95),
            
            sharpe_5th=np.percentile(all_sharpes, 5),
            sharpe_25th=np.percentile(all_sharpes, 25),
            sharpe_50th=np.percentile(all_sharpes, 50),
            sharpe_75th=np.percentile(all_sharpes, 75),
            sharpe_95th=np.percentile(all_sharpes, 95),
            
            drawdown_5th=np.percentile(all_max_drawdowns, 5),
            drawdown_50th=np.percentile(all_max_drawdowns, 50),
            drawdown_95th=np.percentile(all_max_drawdowns, 95),
            
            prob_profitable=np.mean(all_final_returns > 0),
            prob_beat_benchmark=np.mean(all_final_returns > 0.07),  # 7% annual benchmark
            prob_positive_sharpe=np.mean(all_sharpes > 0),
            
            expected_max_drawdown=np.mean(all_max_drawdowns),
            worst_case_drawdown=np.percentile(all_max_drawdowns, 95),
            var_95=np.percentile(all_final_returns, 5),
            cvar_95=np.mean(all_final_returns[all_final_returns <= np.percentile(all_final_returns, 5)]),
            
            passes_validation=False,
            validation_message="",
            
            all_returns=all_final_returns,
            all_sharpes=all_sharpes,
            all_drawdowns=all_max_drawdowns,
            equity_curves=equity_curves
        )
        
        # Run validation
        result = self._validate_result(result)
        
        return result
    
    def simulate_from_returns(
        self,
        daily_returns: pd.Series,
        initial_capital: float = 10000.0,
        iterations: int = None
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation from daily returns series.
        
        Args:
            daily_returns: Series of daily returns
            initial_capital: Starting capital
            iterations: Number of simulations
            
        Returns:
            MonteCarloResult with statistics
        """
        iterations = iterations or self.DEFAULT_ITERATIONS
        
        returns = daily_returns.dropna().values
        
        logger.info(f"Running {iterations} simulations on {len(returns)} daily returns...")
        
        all_final_returns = []
        all_sharpes = []
        all_max_drawdowns = []
        
        for _ in range(iterations):
            # Bootstrap with blocks to preserve some autocorrelation
            block_size = 5
            n_blocks = len(returns) // block_size
            blocks = np.array_split(returns, n_blocks)
            
            # Randomly select blocks with replacement
            selected_blocks = [blocks[np.random.randint(0, len(blocks))] for _ in range(n_blocks)]
            sim_returns = np.concatenate(selected_blocks)
            
            # Calculate metrics
            equity = self._calculate_equity_curve(sim_returns, initial_capital)
            final_return = (equity[-1] - initial_capital) / initial_capital
            sharpe = self._calculate_sharpe(sim_returns)
            max_dd = self._calculate_max_drawdown(equity)
            
            all_final_returns.append(final_return)
            all_sharpes.append(sharpe)
            all_max_drawdowns.append(max_dd)
        
        all_final_returns = np.array(all_final_returns)
        all_sharpes = np.array(all_sharpes)
        all_max_drawdowns = np.array(all_max_drawdowns)
        
        # Original metrics
        original_equity = self._calculate_equity_curve(returns, initial_capital)
        original_return = (original_equity[-1] - initial_capital) / initial_capital
        original_sharpe = self._calculate_sharpe(returns)
        
        result = MonteCarloResult(
            iterations=iterations,
            original_return=original_return,
            original_sharpe=original_sharpe,
            
            return_5th=np.percentile(all_final_returns, 5),
            return_25th=np.percentile(all_final_returns, 25),
            return_50th=np.percentile(all_final_returns, 50),
            return_75th=np.percentile(all_final_returns, 75),
            return_95th=np.percentile(all_final_returns, 95),
            
            sharpe_5th=np.percentile(all_sharpes, 5),
            sharpe_25th=np.percentile(all_sharpes, 25),
            sharpe_50th=np.percentile(all_sharpes, 50),
            sharpe_75th=np.percentile(all_sharpes, 75),
            sharpe_95th=np.percentile(all_sharpes, 95),
            
            drawdown_5th=np.percentile(all_max_drawdowns, 5),
            drawdown_50th=np.percentile(all_max_drawdowns, 50),
            drawdown_95th=np.percentile(all_max_drawdowns, 95),
            
            prob_profitable=np.mean(all_final_returns > 0),
            prob_beat_benchmark=np.mean(all_final_returns > 0.07),
            prob_positive_sharpe=np.mean(all_sharpes > 0),
            
            expected_max_drawdown=np.mean(all_max_drawdowns),
            worst_case_drawdown=np.percentile(all_max_drawdowns, 95),
            var_95=np.percentile(all_final_returns, 5),
            cvar_95=np.mean(all_final_returns[all_final_returns <= np.percentile(all_final_returns, 5)]),
            
            passes_validation=False,
            validation_message="",
            
            all_returns=all_final_returns,
            all_sharpes=all_sharpes,
            all_drawdowns=all_max_drawdowns
        )
        
        result = self._validate_result(result)
        
        return result
    
    def _shuffle_trades(self, returns: np.ndarray) -> np.ndarray:
        """Randomly shuffle trade order"""
        shuffled = returns.copy()
        np.random.shuffle(shuffled)
        return shuffled
    
    def _skip_trades(self, returns: np.ndarray) -> np.ndarray:
        """Randomly skip 5-15% of trades"""
        skip_rate = np.random.uniform(self.SKIP_RATE_MIN, self.SKIP_RATE_MAX)
        mask = np.random.random(len(returns)) > skip_rate
        return returns[mask]
    
    def _bootstrap_trades(self, returns: np.ndarray) -> np.ndarray:
        """Sample trades with replacement"""
        indices = np.random.choice(len(returns), size=len(returns), replace=True)
        return returns[indices]
    
    def _calculate_equity_curve(self, returns: np.ndarray, initial_capital: float) -> np.ndarray:
        """Calculate equity curve from returns"""
        equity = [initial_capital]
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        return np.array(equity)
    
    def _calculate_sharpe(self, returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Calculate Sharpe ratio (annualized)"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        # Assume daily returns, annualize
        sharpe = (mean_return - risk_free) / std_return * np.sqrt(252)
        return sharpe
    
    def _calculate_max_drawdown(self, equity: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return abs(np.min(drawdown))
    
    def _validate_result(self, result: MonteCarloResult) -> MonteCarloResult:
        """
        Validate Monte Carlo result against criteria.
        
        Criteria:
        1. 95% of simulations profitable
        2. Median Sharpe > 0.5
        3. 95th percentile drawdown < 30%
        """
        issues = []
        
        if result.prob_profitable < self.MIN_PROFITABLE_PCT:
            issues.append(f"Only {result.prob_profitable:.1%} profitable (need {self.MIN_PROFITABLE_PCT:.0%})")
        
        if result.sharpe_50th < self.MIN_MEDIAN_SHARPE:
            issues.append(f"Median Sharpe {result.sharpe_50th:.2f} < {self.MIN_MEDIAN_SHARPE}")
        
        if result.drawdown_95th > self.MAX_95_DRAWDOWN:
            issues.append(f"95th percentile drawdown {result.drawdown_95th:.1%} > {self.MAX_95_DRAWDOWN:.0%}")
        
        if issues:
            result.passes_validation = False
            result.validation_message = "FAIL: " + "; ".join(issues)
        else:
            result.passes_validation = True
            result.validation_message = "PASS: Strategy is statistically robust"
        
        return result
    
    def print_result(self, result: MonteCarloResult):
        """Pretty print Monte Carlo result"""
        status = "✅ PASS" if result.passes_validation else "❌ FAIL"
        
        print("\n" + "=" * 60)
        print(f"MONTE CARLO SIMULATION RESULTS ({result.iterations} iterations)")
        print("=" * 60)
        print(f"Validation Status: {status}")
        print(f"Message: {result.validation_message}")
        print("-" * 60)
        
        print("\n📈 RETURN DISTRIBUTION:")
        print(f"  Original Return: {result.original_return:+.1%}")
        print(f"  5th Percentile:  {result.return_5th:+.1%} (worst case)")
        print(f"  25th Percentile: {result.return_25th:+.1%}")
        print(f"  Median (50th):   {result.return_50th:+.1%}")
        print(f"  75th Percentile: {result.return_75th:+.1%}")
        print(f"  95th Percentile: {result.return_95th:+.1%} (best case)")
        
        print("\n📊 SHARPE RATIO DISTRIBUTION:")
        print(f"  Original Sharpe: {result.original_sharpe:.2f}")
        print(f"  5th Percentile:  {result.sharpe_5th:.2f}")
        print(f"  Median (50th):   {result.sharpe_50th:.2f}")
        print(f"  95th Percentile: {result.sharpe_95th:.2f}")
        
        print("\n📉 DRAWDOWN ANALYSIS:")
        print(f"  Expected Max DD: {result.expected_max_drawdown:.1%}")
        print(f"  Median Max DD:   {result.drawdown_50th:.1%}")
        print(f"  Worst Case DD:   {result.worst_case_drawdown:.1%} (95th pctl)")
        
        print("\n🎲 PROBABILITY METRICS:")
        print(f"  Prob. Profitable:     {result.prob_profitable:.1%}")
        print(f"  Prob. Beat Benchmark: {result.prob_beat_benchmark:.1%}")
        print(f"  Prob. Sharpe > 0:     {result.prob_positive_sharpe:.1%}")
        
        print("\n⚠️  RISK METRICS:")
        print(f"  VaR (95%):  {result.var_95:+.1%} (5th percentile return)")
        print(f"  CVaR (95%): {result.cvar_95:+.1%} (avg of worst 5%)")
        
        print("=" * 60)


def quick_monte_carlo(returns: List[float], iterations: int = 1000) -> Dict:
    """
    Quick Monte Carlo analysis for a list of trade returns.
    
    Args:
        returns: List of trade returns (e.g., [0.02, -0.01, 0.03, ...])
        iterations: Number of simulations
        
    Returns:
        Dict with key statistics
    """
    mc = MonteCarloSimulator()
    df = pd.DataFrame({'return': returns})
    result = mc.simulate_from_trades(df, iterations=iterations)
    
    return {
        'passes_validation': result.passes_validation,
        'prob_profitable': result.prob_profitable,
        'median_return': result.return_50th,
        'median_sharpe': result.sharpe_50th,
        'worst_case_return': result.return_5th,
        'worst_case_drawdown': result.worst_case_drawdown,
        'var_95': result.var_95
    }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("MONTE CARLO SIMULATOR")
    print("=" * 60)
    
    # Generate sample trades
    np.random.seed(42)
    
    # Simulate a strategy with 60% win rate, avg win 2%, avg loss 1%
    n_trades = 100
    wins = np.random.normal(0.02, 0.005, int(n_trades * 0.6))
    losses = np.random.normal(-0.01, 0.003, int(n_trades * 0.4))
    sample_returns = np.concatenate([wins, losses])
    np.random.shuffle(sample_returns)
    
    print(f"\nSample Strategy: {n_trades} trades")
    print(f"Win Rate: 60%, Avg Win: 2%, Avg Loss: 1%")
    print(f"Actual Return: {np.sum(sample_returns):.1%}")
    
    # Run Monte Carlo
    mc = MonteCarloSimulator()
    trades_df = pd.DataFrame({'return': sample_returns})
    
    print("\n[1] Running Shuffle Simulation...")
    result_shuffle = mc.simulate_from_trades(trades_df, method='shuffle')
    mc.print_result(result_shuffle)
    
    print("\n[2] Running Skip Simulation...")
    result_skip = mc.simulate_from_trades(trades_df, method='skip')
    mc.print_result(result_skip)
    
    print("\n[3] Running Bootstrap Simulation...")
    result_bootstrap = mc.simulate_from_trades(trades_df, method='bootstrap')
    mc.print_result(result_bootstrap)
    
    # Quick analysis
    print("\n[4] Quick Monte Carlo Check...")
    quick = quick_monte_carlo(list(sample_returns))
    print(f"  Passes Validation: {quick['passes_validation']}")
    print(f"  Prob. Profitable: {quick['prob_profitable']:.1%}")
    print(f"  Median Return: {quick['median_return']:.1%}")
