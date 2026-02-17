"""
Rigorous Simulation Engine
==========================

A comprehensive Monte Carlo simulation framework for validating trading strategies.

Features:
- Monte Carlo simulation with bootstrap resampling
- Realistic execution modeling (slippage, delays, partial fills)
- Walk-forward analysis
- Robustness grading (0-30 scale)
- Strategy comparison

Based on industry best practices:
- 1,000-10,000 simulations for statistical confidence
- 10% skip rate for missed entries
- 30th percentile metrics (conservative)
- B grade (19+) minimum for live trading

Usage:
    from simulation import MonteCarloEngine, SimulationConfig, Trade
    
    config = SimulationConfig(num_simulations=1000)
    engine = MonteCarloEngine(config)
    report = engine.run_simulation(trades, initial_capital=10000)
    print_monte_carlo_report(report)

Author: Trading Bot Arsenal
Created: January 2026
"""

from simulation.monte_carlo_engine import (
    MonteCarloEngine,
    SimulationConfig,
    SimulationResult,
    MonteCarloReport,
    Trade,
    TradeShuffler,
    print_monte_carlo_report,
    save_report_to_json
)

from simulation.realistic_execution import (
    RealisticExecutionModel,
    ExecutionConfig,
    ExecutionResult,
    MarketCondition,
    SlippageModel,
    SkipTradeSimulator
)

from simulation.run_simulation import SimulationRunner

__all__ = [
    # Monte Carlo
    'MonteCarloEngine',
    'SimulationConfig',
    'SimulationResult',
    'MonteCarloReport',
    'Trade',
    'TradeShuffler',
    'print_monte_carlo_report',
    'save_report_to_json',
    
    # Execution
    'RealisticExecutionModel',
    'ExecutionConfig',
    'ExecutionResult',
    'MarketCondition',
    'SlippageModel',
    'SkipTradeSimulator',
    
    # Runner
    'SimulationRunner'
]

__version__ = '1.0.0'
