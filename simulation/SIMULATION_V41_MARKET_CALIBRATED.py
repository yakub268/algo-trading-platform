"""
SIMULATION v4.1 - MARKET CALIBRATED
====================================

Based on:
1. ACTUAL paper trading data (112 trades, Jan 14-27)
2. Wall Street 2026 forecasts (Goldman, Morgan Stanley, JP Morgan)
3. Fed outlook (hold through 2026, 1-2 cuts possible)
4. Crypto outlook (volatile, possible new ATH)

MARKET SCENARIO PROBABILITIES (Feb-July 2026):
Based on consensus forecasts and historical patterns

Source data:
- Goldman Sachs: S&P +12% for 2026
- Morgan Stanley: +14% with volatility
- CFRA: Volatile midterm year
- JP Morgan: Fed on hold at 3.5-3.75%
- Crypto: Wide range $50K-$250K BTC
"""

import os, sys, time, random, signal, json
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from typing import List, Dict
from statistics import median
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# BOT PROFILES - FROM ACTUAL PAPER TRADING (Jan 14-27, 2026)
# =============================================================================

def regress_win_rate(actual_wr, regression_pct=0.15, mean=0.55):
    """Regress extreme win rates toward mean (small sample adjustment)"""
    return actual_wr * (1 - regression_pct) + mean * regression_pct

BOT_PROFILES = {
    # STOCKS - Paper Trading Validated
    'RSI2-MeanReversion': {
        'win_rate': regress_win_rate(0.6364),  # Paper: 63.64%
        'rr_ratio': 0.93,
        'trades_per_week': 8, 'allocation': 0.10,
        'market': 'stocks', 'correlation': 0.70,
        'fee_pct': 0.0015, 'slippage_pct': 0.0015,
        'max_capacity': 50000, 'enabled': True,
    },
    'CumulativeRSI': {
        'win_rate': regress_win_rate(0.5455),  # Paper: 54.55%
        'rr_ratio': 2.52,  # Excellent R:R!
        'trades_per_week': 6, 'allocation': 0.06,
        'market': 'stocks', 'correlation': 0.70,
        'fee_pct': 0.0015, 'slippage_pct': 0.0015,
        'max_capacity': 40000, 'enabled': True,
    },
    'MACD-RSI-Combo': {
        'win_rate': regress_win_rate(0.50),  # Paper: 50%
        'rr_ratio': 1.77,
        'trades_per_week': 5, 'allocation': 0.04,
        'market': 'stocks', 'correlation': 0.70,
        'fee_pct': 0.0015, 'slippage_pct': 0.0015,
        'max_capacity': 40000, 'enabled': True,
    },
    'BollingerSqueeze': {
        'win_rate': regress_win_rate(0.8889, regression_pct=0.25),  # Paper: 88.89%
        'rr_ratio': 1.96,
        'trades_per_week': 4, 'allocation': 0.05,
        'market': 'stocks', 'correlation': 0.70,
        'fee_pct': 0.0015, 'slippage_pct': 0.0015,
        'max_capacity': 40000, 'enabled': True,
    },
    'MTF-RSI': {
        'win_rate': 0.52,  # Estimated
        'rr_ratio': 1.30,
        'trades_per_week': 5, 'allocation': 0.03,
        'market': 'stocks', 'correlation': 0.70,
        'fee_pct': 0.0015, 'slippage_pct': 0.0015,
        'max_capacity': 40000, 'enabled': True,
    },
    'DualMomentum': {
        'win_rate': regress_win_rate(0.50),  # Paper: 50%
        'rr_ratio': 1.62,
        'trades_per_week': 1, 'allocation': 0.06,
        'market': 'stocks', 'correlation': 0.75,
        'fee_pct': 0.0015, 'slippage_pct': 0.0010,
        'max_capacity': 100000, 'enabled': True,
    },
    'SectorRotation': {
        'win_rate': regress_win_rate(0.8333, regression_pct=0.25),  # Paper: 83.33%
        'rr_ratio': 2.50,
        'trades_per_week': 1, 'allocation': 0.06,
        'market': 'stocks', 'correlation': 0.75,
        'fee_pct': 0.0015, 'slippage_pct': 0.0010,
        'max_capacity': 100000, 'enabled': True,
    },
    
    # KALSHI - Paper Trading Validated
    'Kalshi-Fed': {
        'win_rate': regress_win_rate(0.80, regression_pct=0.20),  # Paper: 80%
        'rr_ratio': 1.61,
        'trades_per_week': 2, 'allocation': 0.03,
        'market': 'kalshi', 'correlation': 0.35,
        'fee_pct': 0.12, 'slippage_pct': 0.015,
        'max_capacity': 5000, 'enabled': True,
    },
    'Weather-Edge': {
        'win_rate': regress_win_rate(0.625),  # Paper: 62.5%
        'rr_ratio': 1.37,
        'trades_per_week': 4, 'allocation': 0.02,
        'market': 'kalshi', 'correlation': 0.05,
        'fee_pct': 0.12, 'slippage_pct': 0.015,
        'max_capacity': 3000, 'enabled': True,
    },
    'Sports-Edge': {
        'win_rate': regress_win_rate(0.75, regression_pct=0.20),  # Paper: 75%
        'rr_ratio': 1.58,
        'trades_per_week': 5, 'allocation': 0.02,
        'market': 'kalshi', 'correlation': 0.05,
        'fee_pct': 0.12, 'slippage_pct': 0.012,
        'max_capacity': 5000, 'enabled': True,
    },
    'Sports-Props': {
        'win_rate': 0.55,  # Estimated
        'rr_ratio': 1.20,
        'trades_per_week': 4, 'allocation': 0.01,
        'market': 'kalshi', 'correlation': 0.05,
        'fee_pct': 0.12, 'slippage_pct': 0.012,
        'max_capacity': 3000, 'enabled': True,
    },
    'BoxOffice-Edge': {
        'win_rate': 0.52,  # Estimated
        'rr_ratio': 1.15,
        'trades_per_week': 1, 'allocation': 0.01,
        'market': 'kalshi', 'correlation': 0.10,
        'fee_pct': 0.12, 'slippage_pct': 0.020,
        'max_capacity': 2000, 'enabled': True,
    },
    'Awards-Edge': {
        'win_rate': 0.53,  # Estimated
        'rr_ratio': 1.15,
        'trades_per_week': 1, 'allocation': 0.01,
        'market': 'kalshi', 'correlation': 0.05,
        'fee_pct': 0.12, 'slippage_pct': 0.020,
        'max_capacity': 1500, 'enabled': True,
    },
    'Climate-Edge': {
        'win_rate': 0.58,  # Estimated from Weather-Edge
        'rr_ratio': 1.30,
        'trades_per_week': 2, 'allocation': 0.02,
        'market': 'kalshi', 'correlation': 0.10,
        'fee_pct': 0.12, 'slippage_pct': 0.015,
        'max_capacity': 3000, 'enabled': True,
    },
    'Economic-Edge': {
        'win_rate': 0.60,  # Estimated from Kalshi-Fed
        'rr_ratio': 1.25,
        'trades_per_week': 2, 'allocation': 0.02,
        'market': 'kalshi', 'correlation': 0.40,
        'fee_pct': 0.12, 'slippage_pct': 0.015,
        'max_capacity': 5000, 'enabled': True,
    },
    
    # FOREX - Paper Trading Validated
    'OANDA-Forex': {
        'win_rate': regress_win_rate(0.80, regression_pct=0.30),  # Paper: 80% (5 trades)
        'rr_ratio': 1.98,
        'trades_per_week': 10, 'allocation': 0.04,
        'market': 'forex', 'correlation': 0.30,
        'fee_pct': 0.002, 'slippage_pct': 0.0005,
        'max_capacity': 150000, 'enabled': True,
    },
    'London-Breakout': {
        'win_rate': regress_win_rate(0.6667),  # Paper: 66.67%
        'rr_ratio': 4.32,  # Exceptional R:R!
        'trades_per_week': 4, 'allocation': 0.04,
        'market': 'forex', 'correlation': 0.30,
        'fee_pct': 0.002, 'slippage_pct': 0.0005,
        'max_capacity': 150000, 'enabled': True,
    },
    
    # CRYPTO - Paper Trading Validated
    'FundingRate-Arb': {
        'win_rate': regress_win_rate(0.875, regression_pct=0.20),  # Paper: 87.5%
        'rr_ratio': 1.79,
        'trades_per_week': 15, 'allocation': 0.08,
        'market': 'crypto', 'correlation': 0.50,
        'fee_pct': 0.003, 'slippage_pct': 0.002,
        'max_capacity': 20000, 'enabled': True,
    },
    'Crypto-Arb': {
        'win_rate': regress_win_rate(0.60),  # Paper: 60%
        'rr_ratio': 0.73,  # Low R:R - concerning
        'trades_per_week': 20, 'allocation': 0.02,
        'market': 'crypto', 'correlation': 0.50,
        'fee_pct': 0.004, 'slippage_pct': 0.003,
        'max_capacity': 15000, 'enabled': True,
    },
    'Kalshi-Hourly-Crypto': {
        'win_rate': 0.48, 'rr_ratio': 0.60,
        'trades_per_week': 100, 'allocation': 0.00,
        'market': 'crypto', 'correlation': 0.55,
        'fee_pct': 0.12, 'slippage_pct': 0.010,
        'max_capacity': 10000, 'enabled': False,
    },
    'Alpaca-Crypto-RSI': {
        'win_rate': 0.55,  # Estimated
        'rr_ratio': 1.30,
        'trades_per_week': 60, 'allocation': 0.06,
        'market': 'crypto', 'correlation': 0.55,
        'fee_pct': 0.004, 'slippage_pct': 0.002,
        'max_capacity': 25000, 'enabled': True,
    },
    
    # EVENT-DRIVEN
    'Earnings-PEAD': {
        'win_rate': 0.48, 'rr_ratio': 1.35,
        'trades_per_week': 2, 'allocation': 0.02,
        'market': 'stocks', 'correlation': 0.50,
        'fee_pct': 0.0015, 'slippage_pct': 0.002,
        'max_capacity': 20000, 'enabled': True,
    },
    'FOMC-Trader': {
        'win_rate': 0.55, 'rr_ratio': 1.40,
        'trades_per_week': 1, 'allocation': 0.02,
        'market': 'stocks', 'correlation': 0.55,
        'fee_pct': 0.0015, 'slippage_pct': 0.003,
        'max_capacity': 15000, 'enabled': True,
    },
}

ACTIVE_BOTS = {k: v for k, v in BOT_PROFILES.items() if v.get('enabled', True)}


# =============================================================================
# MARKET SCENARIOS - CALIBRATED TO 2026 WALL STREET FORECASTS
# =============================================================================
# 
# Sources:
# - Goldman Sachs: S&P +12% for 2026, broadening bull
# - Morgan Stanley: +14% with volatility
# - JP Morgan: Fed on hold, growth at 3.3%
# - CFRA: Volatile midterm year
# - Charles Schwab: "Climbing wall of worry", high churn
#
# Feb-July 2026 Scenarios:

class MarketRegime:
    """
    2026-Specific Market Scenarios
    Based on Wall Street forecasts for Feb-July 2026
    """
    REGIMES = {
        # SCENARIO 1: Strong Bull (S&P rallies to +20% YTD by July)
        # Goldman base case + fiscal stimulus from OBBBA
        'strong_bull': {
            'name': 'Strong Bull (GS base case)',
            'description': 'S&P rallies, AI boom continues, fiscal stimulus kicks in',
            'win_rate_modifier': 1.05,  # Stocks outperform
            'rr_modifier': 1.05,
            'market_drift': 0.0004,
            'volatility': 0.12,
            'correlation_multiplier': 0.9,  # Lower correlation in bull
            'position_size_multiplier': 1.00,
            'probability': 0.25,  # 25% chance
            'crypto_modifier': 1.10,  # Crypto benefits from risk-on
        },
        
        # SCENARIO 2: Moderate Bull (S&P +8-12% by July)
        # Consensus case - steady growth, some volatility
        'moderate_bull': {
            'name': 'Moderate Bull (Consensus)',
            'description': 'Steady gains with normal volatility, Fed on hold',
            'win_rate_modifier': 1.02,
            'rr_modifier': 1.00,
            'market_drift': 0.0002,
            'volatility': 0.15,
            'correlation_multiplier': 1.0,
            'position_size_multiplier': 1.00,
            'probability': 0.35,  # 35% chance - most likely
            'crypto_modifier': 1.00,
        },
        
        # SCENARIO 3: Sideways/Volatile (S&P flat to +5%)
        # Midterm election volatility, rotation, "wall of worry"
        'sideways_volatile': {
            'name': 'Sideways Volatile (Schwab/CFRA)',
            'description': 'High churn, rotation, midterm volatility',
            'win_rate_modifier': 0.94,  # Harder to profit in chop
            'rr_modifier': 0.92,
            'market_drift': 0.0,
            'volatility': 0.20,  # Higher VIX
            'correlation_multiplier': 1.1,
            'position_size_multiplier': 0.70,
            'probability': 0.25,  # 25% chance
            'crypto_modifier': 0.90,  # Crypto struggles in uncertainty
        },
        
        # SCENARIO 4: Correction (S&P -10% to -15%)
        # Valuation reset, "less than perfect" economic data
        'correction': {
            'name': 'Correction (Valuation Reset)',
            'description': 'Overvaluation catches up, Fed stays hawkish',
            'win_rate_modifier': 0.85,
            'rr_modifier': 0.82,
            'market_drift': -0.0015,
            'volatility': 0.28,
            'correlation_multiplier': 1.5,
            'position_size_multiplier': 0.40,
            'probability': 0.12,  # 12% chance
            'crypto_modifier': 0.70,  # Crypto sells off harder
        },
        
        # SCENARIO 5: Crash/Crisis (S&P -20%+)
        # Black swan, geopolitical, systemic risk
        'crash': {
            'name': 'Crash (Black Swan)',
            'description': 'Unexpected crisis, VIX spikes to 50+',
            'win_rate_modifier': 0.70,
            'rr_modifier': 0.70,
            'market_drift': -0.006,
            'volatility': 0.45,
            'correlation_multiplier': 1.9,
            'position_size_multiplier': 0.15,
            'probability': 0.03,  # 3% chance
            'crypto_modifier': 0.50,  # Crypto crashes hard
        },
    }
    
    @classmethod
    def get_random_regime(cls) -> str:
        regimes = list(cls.REGIMES.keys())
        probs = [cls.REGIMES[r]['probability'] for r in regimes]
        return random.choices(regimes, weights=probs, k=1)[0]
    
    @classmethod
    def print_scenarios(cls):
        print("\n📊 2026 MARKET SCENARIOS (Feb-July)")
        print("=" * 70)
        for name, r in cls.REGIMES.items():
            print(f"\n{r['name']} - {r['probability']*100:.0f}% probability")
            print(f"   {r['description']}")
            print(f"   WR modifier: {r['win_rate_modifier']:.2f} | Vol: {r['volatility']*100:.0f}%")


@dataclass
class PortfolioConfig:
    starting_capital: float
    regime: str
    num_weeks: int
    seed: int
    strategy_decay_rate: float = 0.004  # 0.4%/week = 10.4%/6mo
    max_drawdown_stop: float = 0.15  # V4: matches 15% max drawdown


@dataclass 
class PortfolioResult:
    config: PortfolioConfig
    ending_capital: float
    peak_capital: float
    max_drawdown_pct: float
    total_trades: int
    total_fees: float
    total_slippage: float
    win_rate: float
    profitable: bool
    total_return_pct: float
    bot_results: Dict[str, Dict]
    weekly_returns: List[float]
    hit_drawdown_stop: bool
    regime_history: List[str]


def simulate_portfolio(config: PortfolioConfig) -> PortfolioResult:
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    regime_data = MarketRegime.REGIMES[config.regime]
    capital = config.starting_capital
    peak_capital = capital
    max_drawdown = 0.0
    total_trades = 0
    total_fees = 0.0
    total_slippage = 0.0
    total_wins = 0
    hit_dd_stop = False
    
    bot_results = {name: {'trades': 0, 'wins': 0, 'pnl': 0.0} for name in ACTIVE_BOTS.keys()}
    weekly_returns = []
    regime_history = [config.regime]
    current_regime = config.regime
    
    for week in range(config.num_weeks):
        week_start_capital = capital
        
        # Check drawdown stop
        current_dd = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
        if current_dd > config.max_drawdown_stop:
            hit_dd_stop = True
            current_regime = 'crash'
            regime_data = MarketRegime.REGIMES[current_regime]
        
        # Regime change probability (10% per week)
        if random.random() < 0.10:
            new_regime = MarketRegime.get_random_regime()
            if new_regime != current_regime:
                current_regime = new_regime
                regime_data = MarketRegime.REGIMES[current_regime]
                regime_history.append(current_regime)
        
        # Decay factor
        decay_factor = max(0.88, 1.0 - (week * config.strategy_decay_rate))
        
        # Market factor for this week
        market_factor = np.random.normal(
            regime_data['market_drift'] * 5, 
            regime_data['volatility'] / np.sqrt(52)
        )
        
        # Extra crash events
        if current_regime == 'crash' and random.random() < 0.15:
            market_factor -= random.uniform(0.03, 0.08)
        
        regime_size_mult = regime_data['position_size_multiplier']
        
        for bot_name, profile in ACTIVE_BOTS.items():
            bot_capital = capital * profile['allocation'] * regime_size_mult
            effective_capital = min(bot_capital, profile['max_capacity'])
            
            if effective_capital < 2.0:
                continue
            
            # Adjust for crypto-specific regime effects
            crypto_mult = regime_data.get('crypto_modifier', 1.0) if profile['market'] == 'crypto' else 1.0
            
            expected_trades = profile['trades_per_week']
            fill_rate = random.uniform(0.72, 0.96)
            actual_trades = max(0, int(np.random.poisson(expected_trades * fill_rate)))
            
            if actual_trades == 0:
                continue
            
            # Calculate win rate with all modifiers
            base_win_rate = profile['win_rate'] * regime_data['win_rate_modifier'] * decay_factor
            if profile['market'] == 'crypto':
                base_win_rate *= crypto_mult
            
            correlation = min(1.0, profile['correlation'] * regime_data['correlation_multiplier'])
            
            # Market factor impact
            if market_factor < 0:
                win_rate_adjustment = market_factor * correlation * 1.4
                base_win_rate = max(0.30, base_win_rate + win_rate_adjustment)
            elif market_factor > 0:
                win_rate_adjustment = market_factor * correlation * 1.0
                base_win_rate = min(0.85, base_win_rate + win_rate_adjustment)
            
            risk_per_trade = effective_capital * random.uniform(0.015, 0.035)
            
            for _ in range(actual_trades):
                total_trades += 1
                bot_results[bot_name]['trades'] += 1
                
                slippage_mult = random.uniform(1.0, 1.4)
                slippage = risk_per_trade * profile['slippage_pct'] * slippage_mult
                total_slippage += slippage
                
                is_win = random.random() < base_win_rate
                
                if is_win:
                    total_wins += 1
                    bot_results[bot_name]['wins'] += 1
                    rr = profile['rr_ratio'] * regime_data.get('rr_modifier', 1.0) * decay_factor
                    win_multiplier = rr * random.uniform(0.80, 1.20)
                    gross_pnl = risk_per_trade * win_multiplier
                    fee = gross_pnl * profile['fee_pct']
                    net_pnl = gross_pnl - fee - slippage
                else:
                    loss_multiplier = random.uniform(0.88, 1.12)
                    gross_pnl = -risk_per_trade * loss_multiplier
                    fee = abs(gross_pnl) * profile['fee_pct'] * 0.2
                    net_pnl = gross_pnl - fee - slippage
                
                total_fees += fee
                bot_results[bot_name]['pnl'] += net_pnl
                capital += net_pnl
                
                if capital < 10:
                    return PortfolioResult(
                        config=config, ending_capital=capital, peak_capital=peak_capital,
                        max_drawdown_pct=100.0, total_trades=total_trades, total_fees=total_fees,
                        total_slippage=total_slippage, win_rate=total_wins/total_trades if total_trades else 0,
                        profitable=False, total_return_pct=-100.0, bot_results=bot_results,
                        weekly_returns=weekly_returns, hit_drawdown_stop=hit_dd_stop,
                        regime_history=regime_history
                    )
        
        week_return = (capital - week_start_capital) / week_start_capital if week_start_capital > 0 else 0
        weekly_returns.append(week_return)
        
        if capital > peak_capital:
            peak_capital = capital
        current_dd = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
        max_drawdown = max(max_drawdown, current_dd)
    
    win_rate = total_wins / total_trades if total_trades > 0 else 0
    total_return = ((capital - config.starting_capital) / config.starting_capital) * 100
    
    return PortfolioResult(
        config=config, ending_capital=capital, peak_capital=peak_capital,
        max_drawdown_pct=max_drawdown * 100, total_trades=total_trades, total_fees=total_fees,
        total_slippage=total_slippage, win_rate=win_rate, profitable=capital > config.starting_capital,
        total_return_pct=total_return, bot_results=bot_results, weekly_returns=weekly_returns,
        hit_drawdown_stop=hit_dd_stop, regime_history=regime_history
    )


def run_batch(configs: List[PortfolioConfig]) -> List[PortfolioResult]:
    return [simulate_portfolio(config) for config in configs]


def generate_configs(batch_size: int, starting_capital: float, num_weeks: int, start_seed: int) -> List[PortfolioConfig]:
    return [PortfolioConfig(starting_capital=starting_capital, regime=MarketRegime.get_random_regime(),
                            num_weeks=num_weeks, seed=start_seed + i) for i in range(batch_size)]


def run_scenario_analysis(starting_capital: float = 500, num_runs: int = 2000, num_weeks: int = 26):
    """Run simulation for each specific scenario"""
    print("\n" + "=" * 75)
    print("🎯 SCENARIO ANALYSIS - How bots perform in each 2026 market condition")
    print("=" * 75)
    
    results = {}
    
    for scenario_name, scenario_data in MarketRegime.REGIMES.items():
        print(f"\n📊 Testing: {scenario_data['name']} ({scenario_data['probability']*100:.0f}% probability)")
        print(f"   {scenario_data['description']}")
        
        configs = [PortfolioConfig(
            starting_capital=starting_capital,
            regime=scenario_name,  # Force this scenario
            num_weeks=num_weeks,
            seed=i + int(time.time())
        ) for i in range(num_runs)]
        
        # Run single-threaded for simplicity
        all_results = [simulate_portfolio(c) for c in configs]
        
        endings = [r.ending_capital for r in all_results]
        profitable = sum(1 for r in all_results if r.profitable)
        broke = sum(1 for r in all_results if r.ending_capital < 10)
        
        results[scenario_name] = {
            'name': scenario_data['name'],
            'probability': scenario_data['probability'],
            'profit_rate': profitable / num_runs,
            'broke_rate': broke / num_runs,
            'median_ending': median(endings),
            'median_return': ((median(endings) - starting_capital) / starting_capital) * 100,
            'p5': np.percentile(endings, 5),
            'p95': np.percentile(endings, 95),
            'max_drawdown_avg': np.mean([r.max_drawdown_pct for r in all_results]),
        }
        
        r = results[scenario_name]
        print(f"   → Profit: {r['profit_rate']:.1%} | Median: ${r['median_ending']:,.0f} ({r['median_return']:+.1f}%)")
    
    # Summary
    print("\n" + "=" * 75)
    print("📋 SCENARIO SUMMARY")
    print("=" * 75)
    print(f"\n{'Scenario':<30}{'Prob':>8}{'Profit':>10}{'Median':>12}{'Return':>10}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{r['name']:<30}{r['probability']*100:>7.0f}%{r['profit_rate']:>9.1%}${r['median_ending']:>11,.0f}{r['median_return']:>+9.1f}%")
    
    # Weighted expected outcome
    weighted_profit = sum(r['profit_rate'] * r['probability'] for r in results.values())
    weighted_return = sum(r['median_return'] * r['probability'] for r in results.values())
    
    print("-" * 70)
    print(f"{'EXPECTED (weighted)':<30}{'100%':>8}{weighted_profit:>9.1%}{'':>12}{weighted_return:>+9.1f}%")
    print("=" * 75)
    
    return results


def run_capital_comparison(capitals=[500, 1000, 2500, 5000, 10000], num_runs=5000, num_weeks=26, num_cores=None):
    if num_cores is None:
        num_cores = cpu_count()
    
    print("\n" + "=" * 75)
    print("📊 V4.1 MARKET-CALIBRATED SIMULATION (2026 Forecasts)")
    print("=" * 75)
    
    MarketRegime.print_scenarios()
    
    print("\n" + "=" * 75)
    print(f"Running {num_runs:,} simulations per capital level...")
    print("=" * 75 + "\n")
    
    results = {}
    
    for capital in capitals:
        print(f"🔄 ${capital:,.0f}...", end=" ", flush=True)
        
        batch_size = max(50, num_runs // num_cores)
        all_configs = []
        seed_counter = (int(time.time() * 1000) + int(capital)) % (2**32 - 1)
        
        remaining = num_runs
        while remaining > 0:
            batch = min(batch_size, remaining)
            configs = generate_configs(batch, capital, num_weeks, seed_counter)
            all_configs.append(configs)
            seed_counter = (seed_counter + batch) % (2**32 - 1)
            remaining -= batch
        
        with Pool(num_cores) as pool:
            results_lists = pool.map(run_batch, all_configs)
        
        all_endings = []
        profitable_count = 0
        broke_count = 0
        all_drawdowns = []
        
        for result_list in results_lists:
            for result in result_list:
                all_endings.append(result.ending_capital)
                all_drawdowns.append(result.max_drawdown_pct)
                if result.profitable: profitable_count += 1
                if result.ending_capital < 10: broke_count += 1
        
        actual_runs = len(all_endings)
        
        results[capital] = {
            'runs': actual_runs,
            'profit_rate': profitable_count / actual_runs,
            'broke_rate': broke_count / actual_runs,
            'median_ending': median(all_endings),
            'avg_ending': sum(all_endings) / actual_runs,
            'median_return_pct': ((median(all_endings) - capital) / capital) * 100,
            'avg_drawdown': np.mean(all_drawdowns),
            'p5': np.percentile(all_endings, 5),
            'p10': np.percentile(all_endings, 10),
            'p25': np.percentile(all_endings, 25),
            'p75': np.percentile(all_endings, 75),
            'p90': np.percentile(all_endings, 90),
            'p95': np.percentile(all_endings, 95),
            'min': min(all_endings),
            'max': max(all_endings),
        }
        
        r = results[capital]
        print(f"✅ {r['profit_rate']:.1%} profit | ${r['median_ending']:,.0f} ({r['median_return_pct']:+.1f}%) | DD: {r['avg_drawdown']:.1f}%")
    
    # Summary
    print("\n" + "=" * 75)
    print("📊 RESULTS - V4.1 Market-Calibrated (Feb-July 2026)")
    print("=" * 75)
    print(f"\n{'Capital':<10}{'Profit':>10}{'Broke':>8}{'Median':>12}{'Return':>10}{'Avg DD':>10}{'5th%':>10}{'95th%':>12}")
    print("-" * 82)
    for capital in capitals:
        r = results[capital]
        print(f"${capital:<9,.0f}{r['profit_rate']:>9.1%}{r['broke_rate']:>8.1%}${r['median_ending']:>11,.0f}{r['median_return_pct']:>+9.1f}%{r['avg_drawdown']:>9.1f}%${r['p5']:>9,.0f}${r['p95']:>11,.0f}")
    
    print("=" * 75)
    
    # Save
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brutal_results')
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, f'v41_market_calibrated_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    save_data = {
        'version': '4.1_market_calibrated',
        'market_scenarios': {k: {kk: vv for kk, vv in v.items() if kk != 'description'} 
                           for k, v in MarketRegime.REGIMES.items()},
        'simulation_results': {str(k): v for k, v in results.items()},
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n📁 Saved: {filepath}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', action='store_true', help='Run capital comparison')
    parser.add_argument('--scenarios', action='store_true', help='Run scenario analysis')
    parser.add_argument('--compare-runs', type=int, default=5000)
    parser.add_argument('--weeks', type=int, default=26)
    parser.add_argument('--capital', type=float, default=500)
    args = parser.parse_args()
    
    if args.scenarios:
        run_scenario_analysis(args.capital, num_runs=2000, num_weeks=args.weeks)
    elif args.compare:
        run_capital_comparison(num_runs=args.compare_runs, num_weeks=args.weeks)
    else:
        print("Use --compare for capital comparison or --scenarios for scenario analysis")


if __name__ == "__main__":
    main()
