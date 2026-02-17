"""
Stress Testing Module for Trading Strategies

Tests strategies against historical market crashes and extreme conditions:
- COVID Crash (Feb-Apr 2020): 34% drop in 23 days
- 2022 Crypto Winter: BTC -77%
- 2010 Flash Crash: 9% drop in minutes
- 2008 GFC: 57% decline
- Simulated stress scenarios

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('StressTest')


class CrisisType(Enum):
    """Types of market crises"""
    COVID_CRASH = "covid_crash"
    CRYPTO_WINTER_2022 = "crypto_winter_2022"
    FLASH_CRASH_2010 = "flash_crash_2010"
    GFC_2008 = "gfc_2008"
    DOT_COM_2000 = "dot_com_2000"
    BLACK_MONDAY_1987 = "black_monday_1987"
    CUSTOM = "custom"


@dataclass
class CrisisPeriod:
    """Definition of a crisis period"""
    name: str
    crisis_type: CrisisType
    start_date: str
    end_date: str
    peak_drawdown: float
    description: str
    asset_class: str = "stocks"


@dataclass
class StressTestResult:
    """Result of stress testing a strategy"""
    strategy_name: str
    crisis_name: str
    crisis_type: CrisisType
    
    # Performance during crisis
    return_during_crisis: float
    max_drawdown: float
    recovery_time_days: Optional[int]
    
    # Trade metrics
    total_trades: int
    win_rate: float
    avg_trade_return: float
    
    # Risk metrics
    sharpe_during_crisis: float
    sortino_during_crisis: float
    
    # Survival assessment
    survived: bool
    survival_score: float  # 0-100
    notes: str


@dataclass
class ComprehensiveStressReport:
    """Complete stress test report across all scenarios"""
    strategy_name: str
    test_date: datetime
    
    # Individual crisis results
    crisis_results: List[StressTestResult]
    
    # Aggregate metrics
    avg_crisis_return: float
    worst_crisis_return: float
    avg_max_drawdown: float
    worst_max_drawdown: float
    overall_survival_rate: float
    
    # Recommendation
    risk_rating: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    recommendation: str
    
    passes_stress_test: bool


class StressTester:
    """
    Comprehensive Stress Testing Framework
    
    Tests strategies against:
    1. Historical crisis periods
    2. Simulated volatility spikes
    3. Liquidity stress scenarios
    4. Execution delays
    """
    
    # Define historical crisis periods
    CRISIS_PERIODS = [
        CrisisPeriod(
            name="COVID Crash",
            crisis_type=CrisisType.COVID_CRASH,
            start_date="2020-02-19",
            end_date="2020-03-23",
            peak_drawdown=0.34,
            description="34% drop in 23 trading days",
            asset_class="stocks"
        ),
        CrisisPeriod(
            name="2022 Crypto Winter",
            crisis_type=CrisisType.CRYPTO_WINTER_2022,
            start_date="2022-01-01",
            end_date="2022-12-31",
            peak_drawdown=0.77,
            description="BTC dropped 77% from ATH",
            asset_class="crypto"
        ),
        CrisisPeriod(
            name="Flash Crash 2010",
            crisis_type=CrisisType.FLASH_CRASH_2010,
            start_date="2010-05-06",
            end_date="2010-05-07",
            peak_drawdown=0.09,
            description="9% drop in minutes",
            asset_class="stocks"
        ),
        CrisisPeriod(
            name="Global Financial Crisis",
            crisis_type=CrisisType.GFC_2008,
            start_date="2008-09-01",
            end_date="2009-03-09",
            peak_drawdown=0.57,
            description="57% decline over 6 months",
            asset_class="stocks"
        ),
        CrisisPeriod(
            name="Dot-Com Crash",
            crisis_type=CrisisType.DOT_COM_2000,
            start_date="2000-03-10",
            end_date="2002-10-09",
            peak_drawdown=0.78,
            description="78% decline in NASDAQ",
            asset_class="stocks"
        ),
    ]
    
    # Survival criteria
    MAX_ACCEPTABLE_DRAWDOWN = 0.30  # 30%
    MIN_SURVIVAL_SCORE = 60  # Out of 100
    
    def __init__(self):
        """Initialize Stress Tester"""
        logger.info("StressTester initialized")
    
    def get_crisis_data(
        self,
        crisis: CrisisPeriod,
        data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Extract data for a specific crisis period.
        
        Args:
            crisis: Crisis period definition
            data: Full historical data (optional, will download if not provided)
            
        Returns:
            DataFrame for crisis period
        """
        if data is None:
            try:
                import yfinance as yf
                
                # Determine symbol based on asset class
                symbol = "SPY" if crisis.asset_class == "stocks" else "BTC-USD"
                
                # Download with buffer
                start = datetime.strptime(crisis.start_date, "%Y-%m-%d") - timedelta(days=30)
                end = datetime.strptime(crisis.end_date, "%Y-%m-%d") + timedelta(days=90)
                
                data = yf.download(symbol, start=start, end=end, progress=False)
                
            except Exception as e:
                logger.error(f"Failed to download data: {e}")
                return pd.DataFrame()
        
        # Filter to crisis period
        crisis_data = data[crisis.start_date:crisis.end_date].copy()
        
        return crisis_data
    
    def simulate_volatility_spike(
        self,
        data: pd.DataFrame,
        multiplier: float = 2.0
    ) -> pd.DataFrame:
        """
        Simulate increased volatility.
        
        Args:
            data: Original price data
            multiplier: Volatility multiplier (2.0 = 2x normal volatility)
            
        Returns:
            Modified data with higher volatility
        """
        data = data.copy()
        close = data['Close'] if 'Close' in data.columns else data['close']
        
        # Calculate returns
        returns = close.pct_change()
        
        # Scale returns by multiplier
        scaled_returns = returns * multiplier
        
        # Reconstruct prices
        initial_price = close.iloc[0]
        new_prices = [initial_price]
        
        for r in scaled_returns.dropna():
            new_prices.append(new_prices[-1] * (1 + r))
        
        # Update close prices
        if 'Close' in data.columns:
            data['Close'] = new_prices[:len(data)]
        else:
            data['close'] = new_prices[:len(data)]
        
        return data
    
    def simulate_slippage(
        self,
        trades: pd.DataFrame,
        slippage_pct: float = 0.005
    ) -> pd.DataFrame:
        """
        Add slippage to trade execution.
        
        Args:
            trades: DataFrame with trade entries/exits
            slippage_pct: Slippage percentage (0.005 = 0.5%)
            
        Returns:
            Trades with slippage applied
        """
        trades = trades.copy()
        
        # Worsen entry prices (buy higher)
        if 'entry_price' in trades.columns:
            trades['entry_price'] = trades['entry_price'] * (1 + slippage_pct)
        
        # Worsen exit prices (sell lower)
        if 'exit_price' in trades.columns:
            trades['exit_price'] = trades['exit_price'] * (1 - slippage_pct)
        
        # Recalculate P&L
        if 'entry_price' in trades.columns and 'exit_price' in trades.columns:
            trades['pnl_pct'] = (trades['exit_price'] - trades['entry_price']) / trades['entry_price']
        
        return trades
    
    def run_strategy_on_crisis(
        self,
        strategy_func: Callable,
        crisis: CrisisPeriod,
        data: pd.DataFrame = None,
        initial_capital: float = 10000.0
    ) -> StressTestResult:
        """
        Run a strategy through a specific crisis period.
        
        Args:
            strategy_func: Strategy function that takes data and returns trades DataFrame
            crisis: Crisis period to test
            data: Optional historical data
            initial_capital: Starting capital
            
        Returns:
            StressTestResult with performance metrics
        """
        # Get crisis data
        crisis_data = self.get_crisis_data(crisis, data)
        
        if crisis_data.empty:
            logger.warning(f"No data available for {crisis.name}")
            return self._create_failed_result(crisis)
        
        # Run strategy
        try:
            trades_df = strategy_func(crisis_data)
        except Exception as e:
            logger.error(f"Strategy failed on {crisis.name}: {e}")
            return self._create_failed_result(crisis)
        
        # Calculate metrics
        if trades_df.empty or len(trades_df) == 0:
            return StressTestResult(
                strategy_name="Unknown",
                crisis_name=crisis.name,
                crisis_type=crisis.crisis_type,
                return_during_crisis=0.0,
                max_drawdown=0.0,
                recovery_time_days=None,
                total_trades=0,
                win_rate=0.0,
                avg_trade_return=0.0,
                sharpe_during_crisis=0.0,
                sortino_during_crisis=0.0,
                survived=True,
                survival_score=50.0,
                notes="No trades during crisis period"
            )
        
        # Calculate returns
        if 'pnl_pct' in trades_df.columns:
            returns = trades_df['pnl_pct'].values
        elif 'return' in trades_df.columns:
            returns = trades_df['return'].values
        else:
            returns = np.array([])
        
        total_return = np.sum(returns) if len(returns) > 0 else 0.0
        
        # Calculate equity curve and drawdown
        equity = [initial_capital]
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        equity = np.array(equity)
        
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = abs(np.min(drawdown))
        
        # Win rate
        wins = np.sum(returns > 0)
        win_rate = wins / len(returns) if len(returns) > 0 else 0.0
        
        # Sharpe/Sortino
        if len(returns) > 1:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            downside = returns[returns < 0]
            sortino = np.mean(returns) / np.std(downside) * np.sqrt(252) if len(downside) > 0 and np.std(downside) > 0 else 0
        else:
            sharpe = 0.0
            sortino = 0.0
        
        # Survival assessment
        survived = max_drawdown < self.MAX_ACCEPTABLE_DRAWDOWN
        survival_score = self._calculate_survival_score(total_return, max_drawdown, win_rate, sharpe)
        
        return StressTestResult(
            strategy_name="Strategy",
            crisis_name=crisis.name,
            crisis_type=crisis.crisis_type,
            return_during_crisis=total_return,
            max_drawdown=max_drawdown,
            recovery_time_days=None,
            total_trades=len(returns),
            win_rate=win_rate,
            avg_trade_return=np.mean(returns) if len(returns) > 0 else 0.0,
            sharpe_during_crisis=sharpe,
            sortino_during_crisis=sortino,
            survived=survived,
            survival_score=survival_score,
            notes=f"{'SURVIVED' if survived else 'FAILED'} - Max DD: {max_drawdown:.1%}"
        )
    
    def _calculate_survival_score(
        self,
        total_return: float,
        max_drawdown: float,
        win_rate: float,
        sharpe: float
    ) -> float:
        """Calculate survival score (0-100)"""
        score = 50.0  # Base score
        
        # Return component (max +/- 20 points)
        if total_return > 0:
            score += min(total_return * 100, 20)
        else:
            score += max(total_return * 100, -20)
        
        # Drawdown component (max -30 points)
        dd_penalty = max_drawdown * 100
        score -= min(dd_penalty, 30)
        
        # Win rate component (max +15 points)
        if win_rate > 0.5:
            score += (win_rate - 0.5) * 30
        
        # Sharpe component (max +15 points)
        if sharpe > 0:
            score += min(sharpe * 5, 15)
        
        return max(0, min(100, score))
    
    def _create_failed_result(self, crisis: CrisisPeriod) -> StressTestResult:
        """Create a failed result for when strategy can't be tested"""
        return StressTestResult(
            strategy_name="Unknown",
            crisis_name=crisis.name,
            crisis_type=crisis.crisis_type,
            return_during_crisis=0.0,
            max_drawdown=1.0,
            recovery_time_days=None,
            total_trades=0,
            win_rate=0.0,
            avg_trade_return=0.0,
            sharpe_during_crisis=0.0,
            sortino_during_crisis=0.0,
            survived=False,
            survival_score=0.0,
            notes="Failed to run strategy"
        )
    
    def run_comprehensive_stress_test(
        self,
        strategy_func: Callable,
        strategy_name: str,
        data: pd.DataFrame = None,
        include_simulated: bool = True
    ) -> ComprehensiveStressReport:
        """
        Run comprehensive stress test across all crisis periods.
        
        Args:
            strategy_func: Strategy function
            strategy_name: Name of strategy
            data: Optional historical data
            include_simulated: Include simulated stress scenarios
            
        Returns:
            ComprehensiveStressReport with all results
        """
        logger.info(f"Running comprehensive stress test for {strategy_name}...")
        
        results = []
        
        # Test against historical crises
        for crisis in self.CRISIS_PERIODS:
            logger.info(f"Testing against {crisis.name}...")
            result = self.run_strategy_on_crisis(strategy_func, crisis, data)
            result.strategy_name = strategy_name
            results.append(result)
        
        # Simulated scenarios
        if include_simulated and data is not None:
            # 2x volatility
            logger.info("Testing against 2x volatility...")
            vol2x_data = self.simulate_volatility_spike(data, 2.0)
            vol2x_result = self.run_strategy_on_crisis(
                strategy_func,
                CrisisPeriod(
                    name="Simulated 2x Volatility",
                    crisis_type=CrisisType.CUSTOM,
                    start_date=str(data.index[0].date()),
                    end_date=str(data.index[-1].date()),
                    peak_drawdown=0.0,
                    description="Simulated 2x normal volatility"
                ),
                vol2x_data
            )
            vol2x_result.strategy_name = strategy_name
            results.append(vol2x_result)
            
            # 3x volatility
            logger.info("Testing against 3x volatility...")
            vol3x_data = self.simulate_volatility_spike(data, 3.0)
            vol3x_result = self.run_strategy_on_crisis(
                strategy_func,
                CrisisPeriod(
                    name="Simulated 3x Volatility",
                    crisis_type=CrisisType.CUSTOM,
                    start_date=str(data.index[0].date()),
                    end_date=str(data.index[-1].date()),
                    peak_drawdown=0.0,
                    description="Simulated 3x normal volatility (extreme)"
                ),
                vol3x_data
            )
            vol3x_result.strategy_name = strategy_name
            results.append(vol3x_result)
        
        # Calculate aggregate metrics
        returns = [r.return_during_crisis for r in results]
        drawdowns = [r.max_drawdown for r in results]
        survival_rates = [r.survived for r in results]
        
        avg_return = np.mean(returns)
        worst_return = np.min(returns)
        avg_dd = np.mean(drawdowns)
        worst_dd = np.max(drawdowns)
        survival_rate = np.mean(survival_rates)
        
        # Determine risk rating
        if worst_dd < 0.15 and survival_rate > 0.9:
            risk_rating = "LOW"
        elif worst_dd < 0.25 and survival_rate > 0.7:
            risk_rating = "MEDIUM"
        elif worst_dd < 0.40 and survival_rate > 0.5:
            risk_rating = "HIGH"
        else:
            risk_rating = "CRITICAL"
        
        # Generate recommendation
        if risk_rating == "LOW":
            recommendation = "Strategy shows strong resilience. Suitable for live deployment."
        elif risk_rating == "MEDIUM":
            recommendation = "Strategy shows acceptable resilience. Consider reduced position sizes."
        elif risk_rating == "HIGH":
            recommendation = "Strategy shows vulnerability. Review risk management before deployment."
        else:
            recommendation = "Strategy fails stress tests. DO NOT deploy without major modifications."
        
        passes = survival_rate >= 0.7 and worst_dd < self.MAX_ACCEPTABLE_DRAWDOWN
        
        return ComprehensiveStressReport(
            strategy_name=strategy_name,
            test_date=datetime.now(),
            crisis_results=results,
            avg_crisis_return=avg_return,
            worst_crisis_return=worst_return,
            avg_max_drawdown=avg_dd,
            worst_max_drawdown=worst_dd,
            overall_survival_rate=survival_rate,
            risk_rating=risk_rating,
            recommendation=recommendation,
            passes_stress_test=passes
        )
    
    def print_report(self, report: ComprehensiveStressReport):
        """Pretty print stress test report"""
        status = "✅ PASS" if report.passes_stress_test else "❌ FAIL"
        
        print("\n" + "=" * 70)
        print(f"COMPREHENSIVE STRESS TEST REPORT: {report.strategy_name}")
        print("=" * 70)
        print(f"Test Date: {report.test_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"Overall Status: {status}")
        print(f"Risk Rating: {report.risk_rating}")
        print("-" * 70)
        
        print("\n📊 CRISIS-BY-CRISIS RESULTS:")
        print("-" * 70)
        print(f"{'Crisis':<30} {'Return':>10} {'Max DD':>10} {'Win Rate':>10} {'Survived':>10}")
        print("-" * 70)
        
        for result in report.crisis_results:
            survived_str = "✅" if result.survived else "❌"
            print(f"{result.crisis_name:<30} {result.return_during_crisis:>+9.1%} {result.max_drawdown:>9.1%} {result.win_rate:>9.1%} {survived_str:>10}")
        
        print("-" * 70)
        
        print("\n📈 AGGREGATE METRICS:")
        print(f"  Average Crisis Return: {report.avg_crisis_return:+.1%}")
        print(f"  Worst Crisis Return:   {report.worst_crisis_return:+.1%}")
        print(f"  Average Max Drawdown:  {report.avg_max_drawdown:.1%}")
        print(f"  Worst Max Drawdown:    {report.worst_max_drawdown:.1%}")
        print(f"  Overall Survival Rate: {report.overall_survival_rate:.1%}")
        
        print("\n💡 RECOMMENDATION:")
        print(f"  {report.recommendation}")
        
        print("=" * 70)


def quick_stress_test(strategy_func: Callable, strategy_name: str = "Strategy") -> Dict:
    """
    Quick stress test for a strategy function.
    
    Args:
        strategy_func: Function that takes DataFrame and returns trades DataFrame
        strategy_name: Name of strategy
        
    Returns:
        Dict with key stress test results
    """
    tester = StressTester()
    report = tester.run_comprehensive_stress_test(strategy_func, strategy_name)
    
    return {
        'passes': report.passes_stress_test,
        'risk_rating': report.risk_rating,
        'survival_rate': report.overall_survival_rate,
        'worst_drawdown': report.worst_max_drawdown,
        'worst_return': report.worst_crisis_return,
        'recommendation': report.recommendation
    }


if __name__ == "__main__":
    print("=" * 60)
    print("STRESS TESTING MODULE")
    print("=" * 60)
    
    # Example: Simple buy-and-hold strategy for testing
    def simple_buy_hold(data: pd.DataFrame) -> pd.DataFrame:
        """Simple buy and hold - for stress test demonstration"""
        close = data['Close'] if 'Close' in data.columns else data['close']
        
        # Single trade: buy at start, sell at end
        entry = close.iloc[0]
        exit_price = close.iloc[-1]
        pnl = (exit_price - entry) / entry
        
        return pd.DataFrame([{
            'entry_price': entry,
            'exit_price': exit_price,
            'pnl_pct': pnl
        }])
    
    # Run stress test
    tester = StressTester()
    
    print("\nRunning stress test on Buy-and-Hold strategy...")
    report = tester.run_comprehensive_stress_test(
        simple_buy_hold,
        "Buy-and-Hold",
        include_simulated=False
    )
    
    tester.print_report(report)
