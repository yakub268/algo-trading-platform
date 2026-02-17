"""
RIGOROUS SIMULATION ENGINE - Monte Carlo Core
==============================================

Based on industry best practices research:
- 1,000-10,000 simulations for statistical confidence
- Bootstrap method with replacement
- 10% skip rate for missed entries
- 30th percentile metrics (conservative)
- Robustness grading (0-30 scale)

Author: Trading Bot Arsenal
Created: January 2026
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import random
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MonteCarloEngine')


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Trade:
    """Single completed trade"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    
    @property
    def is_winner(self) -> bool:
        return self.pnl > 0
    
    @property
    def hold_time(self) -> timedelta:
        return self.exit_time - self.entry_time


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation"""
    num_simulations: int = 1000
    skip_rate: float = 0.10  # 10% of trades randomly skipped
    slippage_pct: float = 0.002  # 0.2% average slippage
    slippage_std: float = 0.001  # Standard deviation
    execution_delay_ms: Tuple[int, int] = (50, 500)  # Min/max delay
    partial_fill_range: Tuple[float, float] = (0.70, 1.0)  # 70-100% fills
    use_replacement: bool = True  # Bootstrap with replacement
    random_seed: Optional[int] = None


@dataclass
class SimulationResult:
    """Results from a single simulation run"""
    simulation_id: int
    total_return: float
    total_return_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    num_trades: int
    num_winners: int
    num_losers: int
    avg_win: float
    avg_loss: float
    max_consecutive_losses: int
    recovery_time_days: float
    equity_curve: List[float]


@dataclass 
class MonteCarloReport:
    """Aggregated Monte Carlo analysis report"""
    # Input parameters
    num_simulations: int
    original_trades: int
    config: SimulationConfig
    
    # Percentile metrics (conservative estimates)
    return_p10: float  # 10th percentile return
    return_p30: float  # 30th percentile return (use this!)
    return_p50: float  # Median return
    return_p70: float
    return_p90: float
    
    drawdown_p10: float
    drawdown_p30: float
    drawdown_p50: float
    drawdown_p70: float
    drawdown_p90: float  # 90th percentile drawdown (worst case)
    
    sharpe_p30: float
    win_rate_p30: float
    profit_factor_p30: float
    
    # Robustness metrics
    robustness_score: int  # 0-30 scale
    robustness_grade: str  # A+, A, B, C, D, F
    
    # Risk of ruin
    probability_of_loss: float
    probability_of_50pct_drawdown: float
    probability_of_ruin: float  # > 80% drawdown
    
    # Recommendations
    go_live_ready: bool
    warnings: List[str]
    
    # All simulation results
    all_results: List[SimulationResult]


class RobustnessGrade(Enum):
    """Robustness grading scale (0-30)"""
    A_PLUS = ("A+", 27, 30, "Excellent - Ready for live trading")
    A = ("A", 23, 26, "Very Good - Ready with caution")
    B = ("B", 19, 22, "Good - Minimum for live trading")
    C = ("C", 15, 18, "Fair - Needs improvement")
    D = ("D", 11, 14, "Poor - Significant issues")
    F = ("F", 0, 10, "Fail - Do not trade")


# =============================================================================
# MONTE CARLO ENGINE
# =============================================================================

class MonteCarloEngine:
    """
    Core Monte Carlo simulation engine.
    
    Implements:
    - Bootstrap resampling with replacement
    - Trade skipping (simulates missed entries)
    - Slippage injection
    - Partial fill simulation
    - Walk-forward validation
    - Robustness scoring
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        if self.config.random_seed:
            np.random.seed(self.config.random_seed)
            random.seed(self.config.random_seed)
    
    def run_simulation(
        self,
        trades: List[Trade],
        initial_capital: float = 10000.0
    ) -> MonteCarloReport:
        """
        Run full Monte Carlo simulation on trade list.
        
        Args:
            trades: List of historical trades from backtest
            initial_capital: Starting capital
            
        Returns:
            MonteCarloReport with all statistics
        """
        logger.info(f"Starting Monte Carlo simulation: {self.config.num_simulations} runs on {len(trades)} trades")
        
        results: List[SimulationResult] = []
        
        # Run simulations (can parallelize for speed)
        for i in range(self.config.num_simulations):
            result = self._run_single_simulation(i, trades, initial_capital)
            results.append(result)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Completed {i + 1}/{self.config.num_simulations} simulations")
        
        # Aggregate results
        report = self._generate_report(trades, results)
        
        logger.info(f"Monte Carlo complete: Grade {report.robustness_grade}, Score {report.robustness_score}/30")
        
        return report
    
    def run_simulation_parallel(
        self,
        trades: List[Trade],
        initial_capital: float = 10000.0,
        max_workers: int = 4
    ) -> MonteCarloReport:
        """
        Run Monte Carlo simulation with parallel processing.
        """
        logger.info(f"Starting parallel Monte Carlo: {self.config.num_simulations} runs, {max_workers} workers")
        
        results: List[SimulationResult] = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._run_single_simulation, i, trades, initial_capital)
                for i in range(self.config.num_simulations)
            ]
            
            for future in as_completed(futures):
                results.append(future.result())
        
        # Sort by simulation_id to maintain order
        results.sort(key=lambda x: x.simulation_id)
        
        return self._generate_report(trades, results)
    
    def _run_single_simulation(
        self,
        sim_id: int,
        trades: List[Trade],
        initial_capital: float
    ) -> SimulationResult:
        """
        Run a single simulation iteration.
        
        Steps:
        1. Bootstrap resample trades (with replacement)
        2. Randomly skip some trades (skip rate)
        3. Apply slippage to remaining trades
        4. Calculate equity curve and metrics
        """
        # Step 1: Bootstrap resample
        if self.config.use_replacement:
            # Sample with replacement (true bootstrap)
            sampled_indices = np.random.choice(
                len(trades), 
                size=len(trades), 
                replace=True
            )
            sampled_trades = [trades[i] for i in sampled_indices]
        else:
            # Shuffle without replacement
            sampled_trades = trades.copy()
            random.shuffle(sampled_trades)
        
        # Step 2: Randomly skip trades
        executed_trades = []
        for trade in sampled_trades:
            if random.random() > self.config.skip_rate:
                executed_trades.append(trade)
        
        # Step 3: Apply slippage
        adjusted_trades = self._apply_slippage(executed_trades)
        
        # Step 4: Calculate metrics
        return self._calculate_simulation_metrics(sim_id, adjusted_trades, initial_capital)
    
    def _apply_slippage(self, trades: List[Trade]) -> List[Trade]:
        """
        Apply random slippage to trade prices.
        
        Slippage is always against the trader:
        - For longs: entry price higher, exit price lower
        - For shorts: entry price lower, exit price higher
        """
        adjusted = []
        
        for trade in trades:
            # Random slippage from normal distribution
            entry_slip = abs(np.random.normal(
                self.config.slippage_pct, 
                self.config.slippage_std
            ))
            exit_slip = abs(np.random.normal(
                self.config.slippage_pct, 
                self.config.slippage_std
            ))
            
            # Partial fill (affects quantity)
            fill_pct = random.uniform(*self.config.partial_fill_range)
            
            if trade.side == 'long':
                new_entry = trade.entry_price * (1 + entry_slip)
                new_exit = trade.exit_price * (1 - exit_slip)
            else:  # short
                new_entry = trade.entry_price * (1 - entry_slip)
                new_exit = trade.exit_price * (1 + exit_slip)
            
            new_quantity = trade.quantity * fill_pct
            new_pnl = (new_exit - new_entry) * new_quantity if trade.side == 'long' else (new_entry - new_exit) * new_quantity
            new_pnl_pct = new_pnl / (new_entry * new_quantity) if new_entry * new_quantity > 0 else 0
            
            adjusted.append(Trade(
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                symbol=trade.symbol,
                side=trade.side,
                entry_price=new_entry,
                exit_price=new_exit,
                quantity=new_quantity,
                pnl=new_pnl,
                pnl_pct=new_pnl_pct
            ))
        
        return adjusted
    
    def _calculate_simulation_metrics(
        self,
        sim_id: int,
        trades: List[Trade],
        initial_capital: float
    ) -> SimulationResult:
        """
        Calculate all metrics for a simulation run.
        """
        if not trades:
            return SimulationResult(
                simulation_id=sim_id,
                total_return=0, total_return_pct=0,
                max_drawdown=0, max_drawdown_pct=0,
                win_rate=0, profit_factor=0,
                sharpe_ratio=0, sortino_ratio=0,
                num_trades=0, num_winners=0, num_losers=0,
                avg_win=0, avg_loss=0,
                max_consecutive_losses=0,
                recovery_time_days=0,
                equity_curve=[initial_capital]
            )
        
        # Build equity curve
        equity = [initial_capital]
        for trade in trades:
            equity.append(equity[-1] + trade.pnl)
        
        # Calculate returns
        total_return = equity[-1] - initial_capital
        total_return_pct = total_return / initial_capital
        
        # Calculate drawdown
        peak = initial_capital
        max_dd = 0
        max_dd_pct = 0
        for value in equity:
            if value > peak:
                peak = value
            dd = peak - value
            dd_pct = dd / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
        
        # Win/loss metrics
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]
        
        win_rate = len(winners) / len(trades) if trades else 0
        
        total_wins = sum(t.pnl for t in winners) if winners else 0
        total_losses = abs(sum(t.pnl for t in losers)) if losers else 0
        
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        avg_win = total_wins / len(winners) if winners else 0
        avg_loss = total_losses / len(losers) if losers else 0
        
        # Calculate Sharpe and Sortino
        returns = [t.pnl_pct for t in trades]
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (avg_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
            
            # Sortino (only downside deviation)
            negative_returns = [r for r in returns if r < 0]
            downside_std = np.std(negative_returns) if negative_returns else 0
            sortino = (avg_return * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
        else:
            sharpe = 0
            sortino = 0
        
        # Consecutive losses
        max_consec_losses = 0
        current_streak = 0
        for trade in trades:
            if not trade.is_winner:
                current_streak += 1
                max_consec_losses = max(max_consec_losses, current_streak)
            else:
                current_streak = 0
        
        # Recovery time (simplified - days to recover from max drawdown)
        recovery_days = 0
        in_drawdown = False
        dd_start = 0
        peak = initial_capital
        for i, value in enumerate(equity):
            if value > peak:
                if in_drawdown:
                    recovery_days = max(recovery_days, i - dd_start)
                    in_drawdown = False
                peak = value
            elif value < peak * 0.95 and not in_drawdown:  # 5% threshold
                in_drawdown = True
                dd_start = i
        
        return SimulationResult(
            simulation_id=sim_id,
            total_return=total_return,
            total_return_pct=total_return_pct,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            num_trades=len(trades),
            num_winners=len(winners),
            num_losers=len(losers),
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_consecutive_losses=max_consec_losses,
            recovery_time_days=recovery_days,
            equity_curve=equity
        )
    
    def _generate_report(
        self,
        original_trades: List[Trade],
        results: List[SimulationResult]
    ) -> MonteCarloReport:
        """
        Generate comprehensive Monte Carlo report with percentiles and grading.
        """
        # Extract metric arrays
        returns = [r.total_return_pct for r in results]
        drawdowns = [r.max_drawdown_pct for r in results]
        sharpes = [r.sharpe_ratio for r in results]
        win_rates = [r.win_rate for r in results]
        profit_factors = [r.profit_factor for r in results]
        
        # Calculate percentiles
        return_p10 = np.percentile(returns, 10)
        return_p30 = np.percentile(returns, 30)
        return_p50 = np.percentile(returns, 50)
        return_p70 = np.percentile(returns, 70)
        return_p90 = np.percentile(returns, 90)
        
        drawdown_p10 = np.percentile(drawdowns, 10)
        drawdown_p30 = np.percentile(drawdowns, 30)
        drawdown_p50 = np.percentile(drawdowns, 50)
        drawdown_p70 = np.percentile(drawdowns, 70)
        drawdown_p90 = np.percentile(drawdowns, 90)
        
        sharpe_p30 = np.percentile(sharpes, 30)
        win_rate_p30 = np.percentile(win_rates, 30)
        pf_p30 = np.percentile([pf for pf in profit_factors if pf != float('inf')], 30) if profit_factors else 0
        
        # Risk probabilities
        prob_loss = len([r for r in returns if r < 0]) / len(returns)
        prob_50_dd = len([d for d in drawdowns if d > 0.50]) / len(drawdowns)
        prob_ruin = len([d for d in drawdowns if d > 0.80]) / len(drawdowns)
        
        # Calculate robustness score (0-30)
        score = self._calculate_robustness_score(
            return_p30, drawdown_p90, sharpe_p30, 
            win_rate_p30, pf_p30, prob_loss, prob_ruin
        )
        
        # Determine grade
        grade = self._get_grade(score)
        
        # Generate warnings
        warnings = []
        if prob_loss > 0.30:
            warnings.append(f"High probability of loss: {prob_loss:.1%}")
        if drawdown_p90 > 0.30:
            warnings.append(f"90th percentile drawdown exceeds 30%: {drawdown_p90:.1%}")
        if sharpe_p30 < 0.5:
            warnings.append(f"30th percentile Sharpe ratio below 0.5: {sharpe_p30:.2f}")
        if win_rate_p30 < 0.40:
            warnings.append(f"30th percentile win rate below 40%: {win_rate_p30:.1%}")
        if prob_ruin > 0.01:
            warnings.append(f"Non-trivial probability of ruin (>80% DD): {prob_ruin:.2%}")
        
        # Go-live decision
        go_live = score >= 19 and prob_ruin < 0.05 and drawdown_p90 < 0.50
        
        return MonteCarloReport(
            num_simulations=len(results),
            original_trades=len(original_trades),
            config=self.config,
            return_p10=return_p10,
            return_p30=return_p30,
            return_p50=return_p50,
            return_p70=return_p70,
            return_p90=return_p90,
            drawdown_p10=drawdown_p10,
            drawdown_p30=drawdown_p30,
            drawdown_p50=drawdown_p50,
            drawdown_p70=drawdown_p70,
            drawdown_p90=drawdown_p90,
            sharpe_p30=sharpe_p30,
            win_rate_p30=win_rate_p30,
            profit_factor_p30=pf_p30,
            robustness_score=score,
            robustness_grade=grade,
            probability_of_loss=prob_loss,
            probability_of_50pct_drawdown=prob_50_dd,
            probability_of_ruin=prob_ruin,
            go_live_ready=go_live,
            warnings=warnings,
            all_results=results
        )
    
    def _calculate_robustness_score(
        self,
        return_p30: float,
        drawdown_p90: float,
        sharpe_p30: float,
        win_rate_p30: float,
        profit_factor_p30: float,
        prob_loss: float,
        prob_ruin: float
    ) -> int:
        """
        Calculate robustness score (0-30 scale).
        
        Scoring breakdown:
        - Return (0-6 points)
        - Drawdown (0-6 points)
        - Sharpe (0-6 points)
        - Win Rate (0-4 points)
        - Profit Factor (0-4 points)
        - Risk Metrics (0-4 points)
        """
        score = 0
        
        # Return score (0-6)
        if return_p30 >= 0.50: score += 6
        elif return_p30 >= 0.30: score += 5
        elif return_p30 >= 0.15: score += 4
        elif return_p30 >= 0.05: score += 3
        elif return_p30 >= 0: score += 2
        elif return_p30 >= -0.10: score += 1
        
        # Drawdown score (0-6, lower is better)
        if drawdown_p90 <= 0.10: score += 6
        elif drawdown_p90 <= 0.15: score += 5
        elif drawdown_p90 <= 0.20: score += 4
        elif drawdown_p90 <= 0.30: score += 3
        elif drawdown_p90 <= 0.40: score += 2
        elif drawdown_p90 <= 0.50: score += 1
        
        # Sharpe score (0-6)
        if sharpe_p30 >= 2.0: score += 6
        elif sharpe_p30 >= 1.5: score += 5
        elif sharpe_p30 >= 1.0: score += 4
        elif sharpe_p30 >= 0.5: score += 3
        elif sharpe_p30 >= 0.25: score += 2
        elif sharpe_p30 >= 0: score += 1
        
        # Win rate score (0-4)
        if win_rate_p30 >= 0.60: score += 4
        elif win_rate_p30 >= 0.50: score += 3
        elif win_rate_p30 >= 0.45: score += 2
        elif win_rate_p30 >= 0.40: score += 1
        
        # Profit factor score (0-4)
        if profit_factor_p30 >= 2.0: score += 4
        elif profit_factor_p30 >= 1.5: score += 3
        elif profit_factor_p30 >= 1.2: score += 2
        elif profit_factor_p30 >= 1.0: score += 1
        
        # Risk metrics score (0-4)
        if prob_loss <= 0.10 and prob_ruin <= 0.001: score += 4
        elif prob_loss <= 0.20 and prob_ruin <= 0.01: score += 3
        elif prob_loss <= 0.30 and prob_ruin <= 0.05: score += 2
        elif prob_loss <= 0.40: score += 1
        
        return min(score, 30)
    
    def _get_grade(self, score: int) -> str:
        """Convert score to letter grade"""
        if score >= 27: return "A+"
        elif score >= 23: return "A"
        elif score >= 19: return "B"
        elif score >= 15: return "C"
        elif score >= 11: return "D"
        else: return "F"


# =============================================================================
# TRADE SHUFFLER (Bootstrap Resampling)
# =============================================================================

class TradeShuffler:
    """
    Bootstrap resampling methods for trade sequences.
    
    Methods:
    - shuffle_with_replacement: True bootstrap
    - shuffle_without_replacement: Permutation test
    - block_shuffle: Preserve some temporal structure
    """
    
    @staticmethod
    def shuffle_with_replacement(trades: List[Trade], n_samples: Optional[int] = None) -> List[Trade]:
        """
        Bootstrap sample with replacement.
        Each trade can appear multiple times or not at all.
        """
        n = n_samples or len(trades)
        indices = np.random.choice(len(trades), size=n, replace=True)
        return [trades[i] for i in indices]
    
    @staticmethod
    def shuffle_without_replacement(trades: List[Trade]) -> List[Trade]:
        """
        Permutation (random shuffle without replacement).
        Each trade appears exactly once but in random order.
        """
        shuffled = trades.copy()
        random.shuffle(shuffled)
        return shuffled
    
    @staticmethod
    def block_shuffle(trades: List[Trade], block_size: int = 10) -> List[Trade]:
        """
        Block bootstrap - shuffle blocks of trades.
        Preserves some temporal autocorrelation.
        """
        n_blocks = len(trades) // block_size
        if n_blocks == 0:
            return TradeShuffler.shuffle_without_replacement(trades)
        
        # Create blocks
        blocks = []
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            blocks.append(trades[start:end])
        
        # Add remainder
        remainder = trades[n_blocks * block_size:]
        if remainder:
            blocks.append(remainder)
        
        # Shuffle blocks
        random.shuffle(blocks)
        
        # Flatten
        return [trade for block in blocks for trade in block]
    
    @staticmethod
    def stratified_shuffle(trades: List[Trade]) -> List[Trade]:
        """
        Stratified shuffle - maintain winner/loser ratio.
        """
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]
        
        random.shuffle(winners)
        random.shuffle(losers)
        
        # Interleave based on original ratio
        result = []
        w_idx, l_idx = 0, 0
        win_ratio = len(winners) / len(trades) if trades else 0.5
        
        for _ in range(len(trades)):
            if random.random() < win_ratio and w_idx < len(winners):
                result.append(winners[w_idx])
                w_idx += 1
            elif l_idx < len(losers):
                result.append(losers[l_idx])
                l_idx += 1
            elif w_idx < len(winners):
                result.append(winners[w_idx])
                w_idx += 1
        
        return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_monte_carlo_report(report: MonteCarloReport):
    """Pretty print Monte Carlo report"""
    print("\n" + "="*60)
    print("MONTE CARLO SIMULATION REPORT")
    print("="*60)
    
    print(f"\n📊 CONFIGURATION")
    print(f"   Simulations: {report.num_simulations:,}")
    print(f"   Original Trades: {report.original_trades:,}")
    print(f"   Skip Rate: {report.config.skip_rate:.0%}")
    print(f"   Slippage: {report.config.slippage_pct:.2%}")
    
    print(f"\n📈 RETURN PERCENTILES")
    print(f"   10th: {report.return_p10:+.1%}")
    print(f"   30th: {report.return_p30:+.1%}  ← Use this (conservative)")
    print(f"   50th: {report.return_p50:+.1%}  (median)")
    print(f"   70th: {report.return_p70:+.1%}")
    print(f"   90th: {report.return_p90:+.1%}")
    
    print(f"\n📉 DRAWDOWN PERCENTILES")
    print(f"   10th: {report.drawdown_p10:.1%}")
    print(f"   30th: {report.drawdown_p30:.1%}")
    print(f"   50th: {report.drawdown_p50:.1%}  (median)")
    print(f"   70th: {report.drawdown_p70:.1%}")
    print(f"   90th: {report.drawdown_p90:.1%}  ← Plan for this (worst case)")
    
    print(f"\n📊 KEY METRICS (30th Percentile)")
    print(f"   Sharpe Ratio: {report.sharpe_p30:.2f}")
    print(f"   Win Rate: {report.win_rate_p30:.1%}")
    print(f"   Profit Factor: {report.profit_factor_p30:.2f}")
    
    print(f"\n⚠️ RISK ANALYSIS")
    print(f"   Probability of Loss: {report.probability_of_loss:.1%}")
    print(f"   Probability of 50% Drawdown: {report.probability_of_50pct_drawdown:.1%}")
    print(f"   Probability of Ruin (>80% DD): {report.probability_of_ruin:.2%}")
    
    print(f"\n🎯 ROBUSTNESS ASSESSMENT")
    print(f"   Score: {report.robustness_score}/30")
    print(f"   Grade: {report.robustness_grade}")
    print(f"   Go-Live Ready: {'✅ YES' if report.go_live_ready else '❌ NO'}")
    
    if report.warnings:
        print(f"\n⚠️ WARNINGS:")
        for warning in report.warnings:
            print(f"   • {warning}")
    
    print("\n" + "="*60)


def save_report_to_json(report: MonteCarloReport, filepath: str):
    """Save Monte Carlo report to JSON file"""
    data = {
        'timestamp': datetime.now().isoformat(),
        'num_simulations': report.num_simulations,
        'original_trades': report.original_trades,
        'config': {
            'skip_rate': report.config.skip_rate,
            'slippage_pct': report.config.slippage_pct,
            'use_replacement': report.config.use_replacement
        },
        'percentiles': {
            'return': {
                'p10': report.return_p10, 'p30': report.return_p30,
                'p50': report.return_p50, 'p70': report.return_p70, 'p90': report.return_p90
            },
            'drawdown': {
                'p10': report.drawdown_p10, 'p30': report.drawdown_p30,
                'p50': report.drawdown_p50, 'p70': report.drawdown_p70, 'p90': report.drawdown_p90
            }
        },
        'key_metrics': {
            'sharpe_p30': report.sharpe_p30,
            'win_rate_p30': report.win_rate_p30,
            'profit_factor_p30': report.profit_factor_p30
        },
        'risk': {
            'probability_of_loss': report.probability_of_loss,
            'probability_of_50pct_drawdown': report.probability_of_50pct_drawdown,
            'probability_of_ruin': report.probability_of_ruin
        },
        'robustness': {
            'score': report.robustness_score,
            'grade': report.robustness_grade,
            'go_live_ready': report.go_live_ready
        },
        'warnings': report.warnings
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Report saved to {filepath}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Generate sample trades for testing
    print("Generating sample trades...")
    
    sample_trades = []
    base_time = datetime.now() - timedelta(days=365)
    
    for i in range(200):
        entry_time = base_time + timedelta(days=i*1.5)
        exit_time = entry_time + timedelta(hours=random.randint(1, 48))
        
        # 55% win rate, realistic distribution
        is_win = random.random() < 0.55
        
        if is_win:
            pnl_pct = random.uniform(0.005, 0.03)  # 0.5% to 3% win
        else:
            pnl_pct = random.uniform(-0.02, -0.005)  # 0.5% to 2% loss
        
        entry_price = 100 + random.uniform(-10, 10)
        exit_price = entry_price * (1 + pnl_pct)
        quantity = random.uniform(10, 100)
        pnl = (exit_price - entry_price) * quantity
        
        sample_trades.append(Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            symbol='TEST',
            side='long',
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=pnl_pct
        ))
    
    # Run Monte Carlo
    config = SimulationConfig(
        num_simulations=1000,
        skip_rate=0.10,
        slippage_pct=0.002,
        random_seed=42
    )
    
    engine = MonteCarloEngine(config)
    report = engine.run_simulation(sample_trades, initial_capital=10000)
    
    # Print report
    print_monte_carlo_report(report)
    
    # Save to file
    save_report_to_json(report, 'monte_carlo_report.json')
