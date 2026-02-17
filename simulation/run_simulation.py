"""
SIMULATION RUNNER - Main Orchestrator
======================================

Runs comprehensive simulations on your trading strategies:
1. Monte Carlo with bootstrap resampling
2. Walk-forward analysis
3. Realistic execution modeling
4. Robustness grading

Usage:
    python run_simulation.py --strategy RSI2 --simulations 1000 --capital 10000

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import random
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.monte_carlo_engine import (
    MonteCarloEngine, 
    SimulationConfig, 
    Trade,
    print_monte_carlo_report,
    save_report_to_json
)
from simulation.realistic_execution import (
    RealisticExecutionModel,
    ExecutionConfig,
    MarketCondition,
    SkipTradeSimulator
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SimulationRunner')


class SimulationRunner:
    """
    Main orchestrator for running comprehensive trading simulations.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        num_simulations: int = 1000,
        skip_rate: float = 0.10,
        slippage_pct: float = 0.002
    ):
        self.initial_capital = initial_capital
        
        # Monte Carlo configuration
        self.mc_config = SimulationConfig(
            num_simulations=num_simulations,
            skip_rate=skip_rate,
            slippage_pct=slippage_pct,
            slippage_std=slippage_pct / 2,
            use_replacement=True
        )
        
        # Execution model
        self.execution_model = RealisticExecutionModel()
        self.skip_simulator = SkipTradeSimulator(base_skip_rate=skip_rate)
        
        # Monte Carlo engine
        self.mc_engine = MonteCarloEngine(self.mc_config)
    
    def run_from_trade_history(
        self,
        trades: List[Trade],
        strategy_name: str = "Unknown"
    ) -> Dict:
        """
        Run simulation on existing trade history.
        """
        logger.info(f"Running simulation for {strategy_name} with {len(trades)} trades")
        
        # Run Monte Carlo
        report = self.mc_engine.run_simulation(trades, self.initial_capital)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"SIMULATION RESULTS: {strategy_name}")
        print(f"{'='*60}")
        print_monte_carlo_report(report)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mc_report_{strategy_name}_{timestamp}.json"
        filepath = os.path.join(os.path.dirname(__file__), filename)
        save_report_to_json(report, filepath)
        
        return {
            'strategy': strategy_name,
            'report': report,
            'filepath': filepath
        }
    
    def run_from_backtest_results(
        self,
        backtest_file: str,
        strategy_name: str = "Unknown"
    ) -> Dict:
        """
        Load trades from a backtest results file and run simulation.
        
        Expected JSON format:
        {
            "trades": [
                {
                    "entry_time": "2024-01-01T10:00:00",
                    "exit_time": "2024-01-01T14:00:00",
                    "symbol": "AAPL",
                    "side": "long",
                    "entry_price": 150.0,
                    "exit_price": 152.0,
                    "quantity": 100,
                    "pnl": 200.0,
                    "pnl_pct": 0.0133
                },
                ...
            ]
        }
        """
        logger.info(f"Loading backtest results from {backtest_file}")
        
        with open(backtest_file, 'r') as f:
            data = json.load(f)
        
        trades = []
        for t in data.get('trades', []):
            trades.append(Trade(
                entry_time=datetime.fromisoformat(t['entry_time']),
                exit_time=datetime.fromisoformat(t['exit_time']),
                symbol=t['symbol'],
                side=t['side'],
                entry_price=t['entry_price'],
                exit_price=t['exit_price'],
                quantity=t['quantity'],
                pnl=t['pnl'],
                pnl_pct=t['pnl_pct']
            ))
        
        return self.run_from_trade_history(trades, strategy_name)
    
    def run_from_database(
        self,
        db_path: str,
        strategy_name: Optional[str] = None
    ) -> Dict:
        """
        Load trades from SQLite database and run simulation.
        """
        import sqlite3
        
        logger.info(f"Loading trades from database: {db_path}")
        
        conn = sqlite3.connect(db_path)
        
        query = "SELECT * FROM trades"
        if strategy_name:
            query += f" WHERE strategy = '{strategy_name}'"
        
        cursor = conn.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        trades = []
        for row in rows:
            data = dict(zip(columns, row))
            trades.append(Trade(
                entry_time=datetime.fromisoformat(data.get('entry_time', data.get('entry_timestamp', ''))),
                exit_time=datetime.fromisoformat(data.get('exit_time', data.get('exit_timestamp', ''))),
                symbol=data.get('symbol', 'UNKNOWN'),
                side=data.get('side', 'long'),
                entry_price=float(data.get('entry_price', 0)),
                exit_price=float(data.get('exit_price', 0)),
                quantity=float(data.get('quantity', data.get('size', 1))),
                pnl=float(data.get('pnl', data.get('profit', 0))),
                pnl_pct=float(data.get('pnl_pct', data.get('return_pct', 0)))
            ))
        
        conn.close()
        
        return self.run_from_trade_history(trades, strategy_name or "Database")
    
    def generate_sample_trades(
        self,
        num_trades: int = 200,
        win_rate: float = 0.55,
        avg_win_pct: float = 0.02,
        avg_loss_pct: float = 0.01
    ) -> List[Trade]:
        """
        Generate sample trades for testing.
        """
        logger.info(f"Generating {num_trades} sample trades (win_rate={win_rate:.0%})")
        
        trades = []
        base_time = datetime.now() - timedelta(days=365)
        
        for i in range(num_trades):
            entry_time = base_time + timedelta(days=i * 1.5)
            exit_time = entry_time + timedelta(hours=random.randint(1, 48))
            
            is_win = random.random() < win_rate
            
            if is_win:
                pnl_pct = random.uniform(avg_win_pct * 0.5, avg_win_pct * 1.5)
            else:
                pnl_pct = -random.uniform(avg_loss_pct * 0.5, avg_loss_pct * 1.5)
            
            entry_price = 100 + random.uniform(-10, 10)
            exit_price = entry_price * (1 + pnl_pct)
            quantity = random.uniform(10, 100)
            pnl = (exit_price - entry_price) * quantity
            
            trades.append(Trade(
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
        
        return trades
    
    def compare_strategies(
        self,
        strategies: Dict[str, List[Trade]]
    ) -> Dict:
        """
        Compare multiple strategies using Monte Carlo analysis.
        """
        results = {}
        
        print("\n" + "="*70)
        print("STRATEGY COMPARISON - MONTE CARLO ANALYSIS")
        print("="*70)
        
        for name, trades in strategies.items():
            logger.info(f"Analyzing strategy: {name}")
            report = self.mc_engine.run_simulation(trades, self.initial_capital)
            results[name] = report
        
        # Print comparison table
        print("\n📊 COMPARISON SUMMARY")
        print("-"*70)
        print(f"{'Strategy':<20} {'Grade':<6} {'Score':<6} {'Return P30':<12} {'DD P90':<10} {'Sharpe':<8}")
        print("-"*70)
        
        for name, report in results.items():
            print(f"{name:<20} {report.robustness_grade:<6} {report.robustness_score:<6} "
                  f"{report.return_p30:>+10.1%} {report.drawdown_p90:>8.1%} {report.sharpe_p30:>7.2f}")
        
        print("-"*70)
        
        # Find best strategy
        best = max(results.items(), key=lambda x: x[1].robustness_score)
        print(f"\n🏆 BEST STRATEGY: {best[0]} (Score: {best[1].robustness_score}/30)")
        
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run Monte Carlo trading simulation')
    
    parser.add_argument('--mode', choices=['sample', 'file', 'db'], default='sample',
                        help='Mode: sample (generate trades), file (load JSON), db (load SQLite)')
    parser.add_argument('--file', type=str, help='Path to backtest results file')
    parser.add_argument('--db', type=str, help='Path to SQLite database')
    parser.add_argument('--strategy', type=str, default='TestStrategy',
                        help='Strategy name')
    parser.add_argument('--simulations', type=int, default=1000,
                        help='Number of Monte Carlo simulations')
    parser.add_argument('--capital', type=float, default=10000,
                        help='Initial capital')
    parser.add_argument('--skip-rate', type=float, default=0.10,
                        help='Trade skip rate (default 10%%)')
    parser.add_argument('--slippage', type=float, default=0.002,
                        help='Average slippage (default 0.2%%)')
    parser.add_argument('--trades', type=int, default=200,
                        help='Number of sample trades to generate')
    parser.add_argument('--win-rate', type=float, default=0.55,
                        help='Sample trade win rate (default 55%%)')
    
    args = parser.parse_args()
    
    # Create runner
    runner = SimulationRunner(
        initial_capital=args.capital,
        num_simulations=args.simulations,
        skip_rate=args.skip_rate,
        slippage_pct=args.slippage
    )
    
    if args.mode == 'sample':
        # Generate and test sample trades
        trades = runner.generate_sample_trades(
            num_trades=args.trades,
            win_rate=args.win_rate
        )
        runner.run_from_trade_history(trades, args.strategy)
        
    elif args.mode == 'file':
        if not args.file:
            print("Error: --file required for file mode")
            sys.exit(1)
        runner.run_from_backtest_results(args.file, args.strategy)
        
    elif args.mode == 'db':
        if not args.db:
            print("Error: --db required for db mode")
            sys.exit(1)
        runner.run_from_database(args.db, args.strategy)


if __name__ == "__main__":
    main()
