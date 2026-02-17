"""
SIMULATION v4.0 - DATA-DRIVEN
=============================

Based on ACTUAL paper trading data (Jan 14-27, 2026)
Source: paper_trading_master.db (112 trades)

METHODOLOGY:
- Use actual win rates from paper trading
- Use actual R:R ratios from paper trading  
- Apply 15% regression to mean (small sample size adjustment)
- Apply regime penalties (data was bull market only)
- Bots without data use conservative estimates

ACTUAL DATA INCORPORATED:
- Overall: 67.9% win rate, $11,283 PnL
- 13 bots with real performance data
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
# V4.0 BOT PROFILES - FROM ACTUAL PAPER TRADING DATA
# =============================================================================
# 
# Adjustments applied:
# 1. Win rates from paper trading with 15% regression to mean
#    (Small sample bias - extreme values regress toward 55%)
# 2. R:R ratios directly from paper trading
# 3. Conservative estimates for bots without data
#
# Formula: adjusted_wr = actual_wr * 0.85 + 0.55 * 0.15

def regress_win_rate(actual_wr, regression_pct=0.15, mean=0.55):
    """Regress extreme win rates toward mean (small sample adjustment)"""
    return actual_wr * (1 - regression_pct) + mean * regression_pct

BOT_PROFILES = {
    # =========================================================================
    # STOCKS - With Paper Trading Data
    # =========================================================================
    'RSI2-MeanReversion': {
        # Paper: 63.64% WR, 0.93 R:R, 11 trades, $536 PnL
        'win_rate': regress_win_rate(0.6364),  # → 0.62
        'rr_ratio': 0.93,  # Actual from paper trading
        'trades_per_week': 8,
        'allocation': 0.10,
        'market': 'stocks',
        'correlation': 0.70,
        'fee_pct': 0.0015,
        'slippage_pct': 0.0015,
        'max_capacity': 50000,
        'enabled': True,
        'data_source': 'paper_trading',
        'paper_trades': 11,
        'paper_pnl': 535.87,
    },
    'CumulativeRSI': {
        # Paper: 54.55% WR, 2.52 R:R, 11 trades, $2,282 PnL
        'win_rate': regress_win_rate(0.5455),  # → 0.55
        'rr_ratio': 2.52,  # Excellent R:R!
        'trades_per_week': 6,
        'allocation': 0.06,  # Increased - top earner
        'market': 'stocks',
        'correlation': 0.70,
        'fee_pct': 0.0015,
        'slippage_pct': 0.0015,
        'max_capacity': 40000,
        'enabled': True,
        'data_source': 'paper_trading',
        'paper_trades': 11,
        'paper_pnl': 2281.53,
    },
    'MACD-RSI-Combo': {
        # Paper: 50% WR, 1.77 R:R, 12 trades, $663 PnL
        'win_rate': regress_win_rate(0.50),  # → 0.51
        'rr_ratio': 1.77,
        'trades_per_week': 5,
        'allocation': 0.04,
        'market': 'stocks',
        'correlation': 0.70,
        'fee_pct': 0.0015,
        'slippage_pct': 0.0015,
        'max_capacity': 40000,
        'enabled': True,
        'data_source': 'paper_trading',
        'paper_trades': 12,
        'paper_pnl': 663.14,
    },
    'BollingerSqueeze': {
        # Paper: 88.89% WR, 1.96 R:R, 9 trades, $1,218 PnL
        # NOTE: 89% WR is likely inflated - heavy regression
        'win_rate': regress_win_rate(0.8889, regression_pct=0.25),  # → 0.80
        'rr_ratio': 1.96,
        'trades_per_week': 4,
        'allocation': 0.05,  # Increased - strong performer
        'market': 'stocks',
        'correlation': 0.70,
        'fee_pct': 0.0015,
        'slippage_pct': 0.0015,
        'max_capacity': 40000,
        'enabled': True,
        'data_source': 'paper_trading',
        'paper_trades': 9,
        'paper_pnl': 1218.20,
    },
    'MTF-RSI': {
        # No paper data - estimate based on similar RSI strategies
        'win_rate': 0.52,
        'rr_ratio': 1.30,
        'trades_per_week': 5,
        'allocation': 0.03,
        'market': 'stocks',
        'correlation': 0.70,
        'fee_pct': 0.0015,
        'slippage_pct': 0.0015,
        'max_capacity': 40000,
        'enabled': True,
        'data_source': 'estimated',
    },
    'DualMomentum': {
        # Paper: 50% WR, 1.62 R:R, 8 trades, $425 PnL
        'win_rate': regress_win_rate(0.50),  # → 0.51
        'rr_ratio': 1.62,
        'trades_per_week': 1,
        'allocation': 0.06,
        'market': 'stocks',
        'correlation': 0.75,
        'fee_pct': 0.0015,
        'slippage_pct': 0.0010,
        'max_capacity': 100000,
        'enabled': True,
        'data_source': 'paper_trading',
        'paper_trades': 8,
        'paper_pnl': 424.88,
    },
    'SectorRotation': {
        # Paper: 83.33% WR, 2.50 R:R, 6 trades, $1,253 PnL
        # NOTE: Small sample, high WR - apply regression
        'win_rate': regress_win_rate(0.8333, regression_pct=0.25),  # → 0.76
        'rr_ratio': 2.50,
        'trades_per_week': 1,
        'allocation': 0.06,  # Increased - excellent R:R
        'market': 'stocks',
        'correlation': 0.75,
        'fee_pct': 0.0015,
        'slippage_pct': 0.0010,
        'max_capacity': 100000,
        'enabled': True,
        'data_source': 'paper_trading',
        'paper_trades': 6,
        'paper_pnl': 1252.58,
    },

    # =========================================================================
    # KALSHI - With Paper Trading Data
    # =========================================================================
    'Kalshi-Fed': {
        # Paper: 80% WR, 1.61 R:R, 10 trades, $0.04 PnL
        # NOTE: Tiny PnL suggests position sizing issue
        'win_rate': regress_win_rate(0.80, regression_pct=0.20),  # → 0.75
        'rr_ratio': 1.61,
        'trades_per_week': 2,
        'allocation': 0.03,
        'market': 'kalshi',
        'correlation': 0.35,
        'fee_pct': 0.12,
        'slippage_pct': 0.015,
        'max_capacity': 5000,
        'enabled': True,
        'data_source': 'paper_trading',
        'paper_trades': 10,
        'paper_pnl': 0.04,
    },
    'Weather-Edge': {
        # Paper: 62.5% WR, 1.37 R:R, 8 trades, $0.02 PnL
        'win_rate': regress_win_rate(0.625),  # → 0.61
        'rr_ratio': 1.37,
        'trades_per_week': 4,
        'allocation': 0.02,
        'market': 'kalshi',
        'correlation': 0.05,
        'fee_pct': 0.12,
        'slippage_pct': 0.015,
        'max_capacity': 3000,
        'enabled': True,
        'data_source': 'paper_trading',
        'paper_trades': 8,
        'paper_pnl': 0.02,
    },
    'Sports-Edge': {
        # Paper: 75% WR, 1.58 R:R, 8 trades, $0.07 PnL
        'win_rate': regress_win_rate(0.75, regression_pct=0.20),  # → 0.71
        'rr_ratio': 1.58,
        'trades_per_week': 5,
        'allocation': 0.02,
        'market': 'kalshi',
        'correlation': 0.05,
        'fee_pct': 0.12,
        'slippage_pct': 0.012,
        'max_capacity': 5000,
        'enabled': True,
        'data_source': 'paper_trading',
        'paper_trades': 8,
        'paper_pnl': 0.07,
    },
    'Sports-Props': {
        # No paper data - estimate based on Sports-Edge
        'win_rate': 0.55,
        'rr_ratio': 1.20,
        'trades_per_week': 4,
        'allocation': 0.01,
        'market': 'kalshi',
        'correlation': 0.05,
        'fee_pct': 0.12,
        'slippage_pct': 0.012,
        'max_capacity': 3000,
        'enabled': True,
        'data_source': 'estimated',
    },
    'BoxOffice-Edge': {
        # No paper data - conservative estimate
        'win_rate': 0.52,
        'rr_ratio': 1.15,
        'trades_per_week': 1,
        'allocation': 0.01,
        'market': 'kalshi',
        'correlation': 0.10,
        'fee_pct': 0.12,
        'slippage_pct': 0.020,
        'max_capacity': 2000,
        'enabled': True,
        'data_source': 'estimated',
    },
    'Awards-Edge': {
        # No paper data - conservative estimate
        'win_rate': 0.53,
        'rr_ratio': 1.15,
        'trades_per_week': 1,
        'allocation': 0.01,
        'market': 'kalshi',
        'correlation': 0.05,
        'fee_pct': 0.12,
        'slippage_pct': 0.020,
        'max_capacity': 1500,
        'enabled': True,
        'data_source': 'estimated',
    },
    'Climate-Edge': {
        # No paper data - estimate based on Weather-Edge
        'win_rate': 0.58,
        'rr_ratio': 1.30,
        'trades_per_week': 2,
        'allocation': 0.02,
        'market': 'kalshi',
        'correlation': 0.10,
        'fee_pct': 0.12,
        'slippage_pct': 0.015,
        'max_capacity': 3000,
        'enabled': True,
        'data_source': 'estimated',
    },
    'Economic-Edge': {
        # No paper data - estimate based on Kalshi-Fed
        'win_rate': 0.60,
        'rr_ratio': 1.25,
        'trades_per_week': 2,
        'allocation': 0.02,
        'market': 'kalshi',
        'correlation': 0.40,
        'fee_pct': 0.12,
        'slippage_pct': 0.015,
        'max_capacity': 5000,
        'enabled': True,
        'data_source': 'estimated',
    },

    # =========================================================================
    # FOREX - With Paper Trading Data
    # =========================================================================
    'OANDA-Forex': {
        # Paper: 80% WR, 1.98 R:R, 5 trades, $0.07 PnL
        # NOTE: Only 5 trades - heavy regression
        'win_rate': regress_win_rate(0.80, regression_pct=0.30),  # → 0.725
        'rr_ratio': 1.98,
        'trades_per_week': 10,
        'allocation': 0.04,
        'market': 'forex',
        'correlation': 0.30,
        'fee_pct': 0.002,
        'slippage_pct': 0.0005,
        'max_capacity': 150000,
        'enabled': True,
        'data_source': 'paper_trading',
        'paper_trades': 5,
        'paper_pnl': 0.07,
    },
    'London-Breakout': {
        # Paper: 66.67% WR, 4.32 R:R, 6 trades, $0.06 PnL
        # NOTE: 4.32 R:R is exceptional!
        'win_rate': regress_win_rate(0.6667),  # → 0.65
        'rr_ratio': 4.32,  # Keep actual - this is the edge
        'trades_per_week': 4,
        'allocation': 0.04,  # Increased - great R:R
        'market': 'forex',
        'correlation': 0.30,
        'fee_pct': 0.002,
        'slippage_pct': 0.0005,
        'max_capacity': 150000,
        'enabled': True,
        'data_source': 'paper_trading',
        'paper_trades': 6,
        'paper_pnl': 0.06,
    },

    # =========================================================================
    # CRYPTO - With Paper Trading Data
    # =========================================================================
    'FundingRate-Arb': {
        # Paper: 87.5% WR, 1.79 R:R, 8 trades, $4,795 PnL
        # TOP PERFORMER - 42% of all profits!
        'win_rate': regress_win_rate(0.875, regression_pct=0.20),  # → 0.81
        'rr_ratio': 1.79,
        'trades_per_week': 15,
        'allocation': 0.08,  # Increased - best performer
        'market': 'crypto',
        'correlation': 0.50,
        'fee_pct': 0.003,
        'slippage_pct': 0.002,
        'max_capacity': 20000,
        'enabled': True,
        'data_source': 'paper_trading',
        'paper_trades': 8,
        'paper_pnl': 4795.37,
    },
    'Crypto-Arb': {
        # Paper: 60% WR, 0.73 R:R, 10 trades, $111 PnL
        # NOTE: Low R:R is concerning
        'win_rate': regress_win_rate(0.60),  # → 0.59
        'rr_ratio': 0.73,  # Below 1.0 - needs monitoring
        'trades_per_week': 20,
        'allocation': 0.02,  # Reduced - poor R:R
        'market': 'crypto',
        'correlation': 0.50,
        'fee_pct': 0.004,
        'slippage_pct': 0.003,
        'max_capacity': 15000,
        'enabled': True,
        'data_source': 'paper_trading',
        'paper_trades': 10,
        'paper_pnl': 111.30,
    },
    'Kalshi-Hourly-Crypto': {
        # DISABLED - negative EV from fee structure
        'win_rate': 0.48,
        'rr_ratio': 0.60,
        'trades_per_week': 100,
        'allocation': 0.00,
        'market': 'crypto',
        'correlation': 0.55,
        'fee_pct': 0.12,
        'slippage_pct': 0.010,
        'max_capacity': 10000,
        'enabled': False,
        'data_source': 'disabled',
    },
    'Alpaca-Crypto-RSI': {
        # No paper data - estimate based on RSI2 and Crypto-Arb
        'win_rate': 0.55,
        'rr_ratio': 1.30,
        'trades_per_week': 60,
        'allocation': 0.06,
        'market': 'crypto',
        'correlation': 0.55,
        'fee_pct': 0.004,
        'slippage_pct': 0.002,
        'max_capacity': 25000,
        'enabled': True,
        'data_source': 'estimated',
    },

    # =========================================================================
    # EVENT-DRIVEN - No Paper Data
    # =========================================================================
    'Earnings-PEAD': {
        # No paper data - conservative estimate
        'win_rate': 0.48,
        'rr_ratio': 1.35,
        'trades_per_week': 2,
        'allocation': 0.02,
        'market': 'stocks',
        'correlation': 0.50,
        'fee_pct': 0.0015,
        'slippage_pct': 0.002,
        'max_capacity': 20000,
        'enabled': True,
        'data_source': 'estimated',
    },
    'FOMC-Trader': {
        # No paper data - estimate based on Kalshi-Fed
        'win_rate': 0.55,
        'rr_ratio': 1.40,
        'trades_per_week': 1,
        'allocation': 0.02,
        'market': 'stocks',
        'correlation': 0.55,
        'fee_pct': 0.0015,
        'slippage_pct': 0.003,
        'max_capacity': 15000,
        'enabled': True,
        'data_source': 'estimated',
    },
}

ACTIVE_BOTS = {k: v for k, v in BOT_PROFILES.items() if v.get('enabled', True)}

# Count data sources
paper_bots = sum(1 for b in ACTIVE_BOTS.values() if b.get('data_source') == 'paper_trading')
estimated_bots = sum(1 for b in ACTIVE_BOTS.values() if b.get('data_source') == 'estimated')


class MarketRegime:
    """
    V4.0: Calibrated for bull market data
    Paper trading was during bull market (Jan 14-27, 2026)
    Apply appropriate penalties for other regimes
    """
    REGIMES = {
        'bull': {
            # This matches paper trading conditions
            'win_rate_modifier': 1.00,  # Baseline - paper data is from bull
            'rr_modifier': 1.00,
            'market_drift': 0.0002,
            'volatility': 0.14,
            'correlation_multiplier': 1.0,
            'position_size_multiplier': 1.00,
            'probability': 0.30
        },
        'bull_volatile': {
            'win_rate_modifier': 0.96,
            'rr_modifier': 0.97,
            'market_drift': 0.0001,
            'volatility': 0.20,
            'correlation_multiplier': 1.15,
            'position_size_multiplier': 0.80,
            'probability': 0.20
        },
        'sideways': {
            'win_rate_modifier': 0.92,  # Expect 8% lower WR in sideways
            'rr_modifier': 0.90,
            'market_drift': 0.0,
            'volatility': 0.16,
            'correlation_multiplier': 1.0,
            'position_size_multiplier': 0.60,
            'probability': 0.25
        },
        'correction': {
            'win_rate_modifier': 0.85,  # 15% lower WR in correction
            'rr_modifier': 0.82,
            'market_drift': -0.002,
            'volatility': 0.28,
            'correlation_multiplier': 1.45,
            'position_size_multiplier': 0.35,
            'probability': 0.15
        },
        'crash': {
            'win_rate_modifier': 0.70,  # 30% lower WR in crash
            'rr_modifier': 0.72,
            'market_drift': -0.008,
            'volatility': 0.42,
            'correlation_multiplier': 1.85,
            'position_size_multiplier': 0.15,
            'probability': 0.10
        }
    }
    
    @classmethod
    def get_random_regime(cls) -> str:
        regimes = list(cls.REGIMES.keys())
        probs = [cls.REGIMES[r]['probability'] for r in regimes]
        return random.choices(regimes, weights=probs, k=1)[0]


@dataclass
class PortfolioConfig:
    starting_capital: float
    regime: str
    num_weeks: int
    seed: int
    strategy_decay_rate: float = 0.004  # 0.4%/week = 10.4%/6mo (conservative)
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


def simulate_portfolio(config: PortfolioConfig) -> PortfolioResult:
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    regime = MarketRegime.REGIMES[config.regime]
    capital = config.starting_capital
    peak_capital = capital
    max_drawdown = 0.0
    total_trades = 0
    total_fees = 0.0
    total_slippage = 0.0
    total_wins = 0
    hit_dd_stop = False
    
    bot_results = {name: {'trades': 0, 'wins': 0, 'pnl': 0.0, 'fees': 0.0, 'slippage': 0.0} for name in ACTIVE_BOTS.keys()}
    weekly_returns = []
    current_regime = config.regime
    
    for week in range(config.num_weeks):
        week_start_capital = capital
        current_dd = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
        if current_dd > config.max_drawdown_stop:
            hit_dd_stop = True
            regime = MarketRegime.REGIMES['crash']
        
        if random.random() < 0.12:
            new_regime = MarketRegime.get_random_regime()
            if new_regime != current_regime:
                current_regime = new_regime
                regime = MarketRegime.REGIMES[current_regime]
        
        decay_factor = max(0.88, 1.0 - (week * config.strategy_decay_rate))
        market_factor = np.random.normal(regime['market_drift'] * 5, regime['volatility'] / np.sqrt(52))
        
        if current_regime == 'crash' and random.random() < 0.12:
            market_factor -= random.uniform(0.02, 0.06)
        
        regime_size_mult = regime['position_size_multiplier']
        
        for bot_name, profile in ACTIVE_BOTS.items():
            bot_capital = capital * profile['allocation'] * regime_size_mult
            effective_capital = min(bot_capital, profile['max_capacity'])
            
            if effective_capital < 2.0:
                continue
            
            expected_trades = profile['trades_per_week']
            fill_rate = random.uniform(0.72, 0.96)
            actual_trades = max(0, int(np.random.poisson(expected_trades * fill_rate)))
            
            if actual_trades == 0:
                continue
            
            # Use paper trading win rate adjusted for regime
            base_win_rate = profile['win_rate'] * regime['win_rate_modifier'] * decay_factor
            correlation = min(1.0, profile['correlation'] * regime['correlation_multiplier'])
            
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
                bot_results[bot_name]['slippage'] += slippage
                
                is_win = random.random() < base_win_rate
                
                if is_win:
                    total_wins += 1
                    bot_results[bot_name]['wins'] += 1
                    # Use actual R:R from paper trading
                    rr = profile['rr_ratio'] * regime.get('rr_modifier', 1.0) * decay_factor
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
                bot_results[bot_name]['fees'] += fee
                bot_results[bot_name]['pnl'] += net_pnl
                capital += net_pnl
                
                if capital < 10:
                    return PortfolioResult(
                        config=config, ending_capital=capital, peak_capital=peak_capital,
                        max_drawdown_pct=100.0, total_trades=total_trades, total_fees=total_fees,
                        total_slippage=total_slippage, win_rate=total_wins / total_trades if total_trades > 0 else 0,
                        profitable=False, total_return_pct=-100.0, bot_results=bot_results,
                        weekly_returns=weekly_returns, hit_drawdown_stop=hit_dd_stop
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
        hit_drawdown_stop=hit_dd_stop
    )


def run_batch(configs: List[PortfolioConfig]) -> List[PortfolioResult]:
    return [simulate_portfolio(config) for config in configs]


def generate_configs(batch_size: int, starting_capital: float, num_weeks: int, start_seed: int) -> List[PortfolioConfig]:
    return [PortfolioConfig(starting_capital=starting_capital, regime=MarketRegime.get_random_regime(),
                            num_weeks=num_weeks, seed=start_seed + i) for i in range(batch_size)]


def run_capital_comparison(capitals=[500, 1000, 2500, 5000, 10000], num_runs=5000, num_weeks=26, num_cores=None):
    if num_cores is None:
        num_cores = cpu_count()
    
    print("\n" + "=" * 75)
    print("📊 SIMULATION v4.0 - DATA-DRIVEN (Using Real Paper Trading)")
    print("=" * 75)
    print(f"Data source: paper_trading_master.db (112 trades, Jan 14-27)")
    print(f"Bots with real data: {paper_bots} | Estimated: {estimated_bots}")
    print(f"Capitals: {', '.join([f'${c:,.0f}' for c in capitals])}")
    print(f"Runs: {num_runs:,} | Weeks: {num_weeks} | Cores: {num_cores}")
    print("=" * 75)
    
    # Show data-backed win rates
    print("\n📋 Win Rates Used (from paper trading):")
    for name, profile in sorted(ACTIVE_BOTS.items(), key=lambda x: x[1]['win_rate'], reverse=True)[:8]:
        source = profile.get('data_source', 'unknown')
        marker = "✅" if source == 'paper_trading' else "📊"
        print(f"   {marker} {name}: {profile['win_rate']:.1%} WR, {profile['rr_ratio']:.2f} R:R")
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
        
        for result_list in results_lists:
            for result in result_list:
                all_endings.append(result.ending_capital)
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
        print(f"✅ {r['profit_rate']:.1%} profit | {r['broke_rate']:.1%} broke | ${r['median_ending']:,.0f} ({r['median_return_pct']:+.1f}%)")
    
    # Summary
    print("\n" + "=" * 75)
    print("📊 RESULTS SUMMARY - V4.0 DATA-DRIVEN")
    print("=" * 75)
    print(f"\n{'Capital':<10}{'Profit':>10}{'Broke':>10}{'Median':>12}{'Return':>10}{'5th %':>10}{'95th %':>12}")
    print("-" * 74)
    for capital in capitals:
        r = results[capital]
        print(f"${capital:<9,.0f}{r['profit_rate']:>9.1%}{r['broke_rate']:>10.1%}${r['median_ending']:>11,.0f}{r['median_return_pct']:>+9.1f}%${r['p5']:>9,.0f}${r['p95']:>11,.0f}")
    
    # Comparison with paper trading
    print("\n" + "=" * 75)
    print("📋 COMPARISON: Simulation vs Paper Trading Reality")
    print("=" * 75)
    print(f"Paper Trading (2 weeks):  67.9% win rate, +$11,283 PnL")
    avg_profit_rate = sum(r['profit_rate'] for r in results.values()) / len(results)
    print(f"V4.0 Simulation (26 wks): {avg_profit_rate:.1%} profit rate over full period")
    print("\nNote: Paper trading was bull market only. Simulation includes")
    print("      corrections (15%) and crashes (10%) which reduce overall profitability.")
    print("=" * 75)
    
    # Save
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brutal_results')
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, f'v40_datadriven_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    save_data = {
        'version': '4.0_data_driven',
        'data_source': 'paper_trading_master.db',
        'paper_trades': 112,
        'paper_period': 'Jan 14-27, 2026',
        'paper_win_rate': 67.86,
        'paper_pnl': 11283.14,
        'bots_with_real_data': paper_bots,
        'bots_estimated': estimated_bots,
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
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--compare-runs', type=int, default=5000)
    parser.add_argument('--weeks', type=int, default=26)
    args = parser.parse_args()
    
    if args.compare:
        run_capital_comparison(num_runs=args.compare_runs, num_weeks=args.weeks)
    else:
        print("Use --compare to run capital comparison")


if __name__ == "__main__":
    main()
