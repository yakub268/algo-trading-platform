"""
Trading Bot Configuration

Centralized configuration for all trading components.
Edit this file to customize your trading parameters.

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


# =============================================================================
# ACCOUNT CONFIGURATION
# =============================================================================

@dataclass
class AccountConfig:
    """Account and capital settings"""
    starting_balance: float = 500.0
    paper_mode: bool = True  # Default to PAPER mode (safe default)
    
    # Allocation percentages (must sum to 1.0)
    allocation: Dict[str, float] = field(default_factory=lambda: {
        'crypto_momentum': 0.40,       # $200 - Primary (Alpaca crypto)
        'prediction_markets': 0.30,    # $150 - Edge detection (Kalshi)
        'forex': 0.20,                 # $100 - London Breakout (OANDA)
        'cash_reserve': 0.10,          # $50 - Safety buffer
    })


# =============================================================================
# RSI-2 STRATEGY CONFIGURATION (Research-Validated)
# =============================================================================

@dataclass
class RSI2Config:
    """RSI-2 Mean Reversion parameters"""
    # Core parameters
    rsi_period: int = 2
    rsi_oversold: int = 10             # Widened from 5 to 10 for more signals (76% vs 81% win rate tradeoff)
    rsi_oversold_streak: int = 10      # Threshold for consecutive counting
    rsi_overbought: int = 90
    consecutive_bars: int = 2          # Reduced from 3 to 2 for faster entries
    
    # Trend filter
    sma_trend: int = 200               # 200 SMA filter (research: improves win rate 76%→81%)
    sma_exit: int = 5                  # Exit when price > 5 SMA
    
    # Stops (research: ATR-based outperforms fixed %)
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0   # 2x ATR stop loss
    fallback_stop_pct: float = 0.03    # 3% if ATR unavailable
    
    # Volume confirmation
    volume_threshold: float = 1.2      # Volume > 1.2x average
    
    # Symbols to trade
    symbols: List[str] = field(default_factory=lambda: ['SPY', 'QQQ', 'IWM'])


# =============================================================================
# DUAL MOMENTUM CONFIGURATION
# =============================================================================

@dataclass
class DualMomentumConfig:
    """Dual Momentum parameters"""
    us_equity: str = "SPY"
    intl_equity: str = "EFA"
    bonds: str = "AGG"

    lookback_months: int = 12          # 12-month momentum
    tbill_rate: float = 0.045          # Current T-bill rate (~4.5%)

    # Rebalance on first 3 days of month
    rebalance_days: List[int] = field(default_factory=lambda: [1, 2, 3])


# =============================================================================
# CRYPTO DCA CONFIGURATION
# =============================================================================

@dataclass
class CryptoDCAConfig:
    """Crypto DCA Accumulation Strategy Configuration"""
    # Tokens to accumulate
    tokens: List[str] = field(default_factory=lambda: ['XRP', 'HBAR', 'XLM'])

    # Total new capital to deploy
    total_capital: float = 500.0

    # Allocation per token (new money only - user has existing XRP)
    allocation: Dict[str, float] = field(default_factory=lambda: {
        'XRP': 0.0,    # Already holding, just manage stop-loss
        'HBAR': 0.60,  # $300 - best risk/reward at current levels
        'XLM': 0.40,   # $200 - oversold bounce play
    })

    # Entry zones (buy when price falls into these ranges)
    entry_zones: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'XRP': {'aggressive_low': 1.50, 'aggressive_high': 1.90, 'conservative_above': 2.02},
        'HBAR': {'aggressive_low': 0.09, 'aggressive_high': 0.12, 'conservative_above': 0.13},
        'XLM': {'aggressive_low': 0.18, 'aggressive_high': 0.21, 'conservative_above': 0.24},
    })

    # Stop loss levels
    stop_losses: Dict[str, float] = field(default_factory=lambda: {
        'XRP': 1.50,
        'HBAR': 0.085,
        'XLM': 0.17,
    })

    # Target prices (for take-profit alerts)
    targets: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'XRP': {'conservative': 3.00, 'moderate': 4.00, 'aggressive': 8.00},
        'HBAR': {'conservative': 0.20, 'moderate': 0.45, 'aggressive': 0.80},
        'XLM': {'conservative': 0.40, 'moderate': 0.60, 'aggressive': 1.00},
    })

    # DCA settings
    dca_tranches: int = 3  # Split buys into 3 tranches
    min_buy_usd: float = 10.0  # Minimum buy size

    # Risk settings
    max_position_pct: float = 0.40  # Max 40% in any single token

    # Scan interval
    scan_interval_minutes: int = 15


# =============================================================================
# RISK MANAGEMENT CONFIGURATION (Research-Validated)
# =============================================================================

@dataclass
class RiskConfig:
    """Risk management parameters"""
    # Position sizing
    base_risk_per_trade: float = 0.01  # 1% risk per trade
    max_risk_per_trade: float = 0.02   # 2% maximum
    kelly_fraction: float = 0.25       # Quarter Kelly (research: safer than full)
    max_position_pct: float = 0.15     # 15% max single position
    max_positions: int = 6
    
    # VIX-based scaling (research: NOT binary, use scaling)
    vix_scaling: Dict[str, float] = field(default_factory=lambda: {
        'low': 1.0,        # VIX < 15: 100%
        'normal': 1.0,     # VIX 15-20: 100%
        'elevated': 0.75,  # VIX 20-25: 75%
        'high': 0.50,      # VIX 25-30: 50%
        'extreme': 0.25,   # VIX 30-40: 25%
        'crisis': 0.0,     # VIX > 40: 0% (halt)
    })
    
    # Circuit breakers (V4: Updated for $500 capital)
    daily_loss_limit: float = 0.03     # V4: 3% daily ($15)
    weekly_loss_limit: float = 0.07    # V4: 7% weekly ($35)
    monthly_loss_limit: float = 0.15   # V4: 15% monthly ($67.50)
    max_drawdown: float = 0.15         # V4: 15% max drawdown
    consecutive_losses_warning: int = 3
    consecutive_losses_halt: int = 5
    
    # Drawdown position scaling (V4: tightened to match 15% max)
    drawdown_scaling: Dict[float, float] = field(default_factory=lambda: {
        0.03: 0.90,   # 3% drawdown → 90% size
        0.06: 0.75,   # 6% drawdown → 75% size
        0.10: 0.50,   # 10% drawdown → 50% size
        0.13: 0.25,   # 13% drawdown → 25% size
        0.15: 0.00,   # 15% drawdown → halt
    })
    
    # Sector limits
    sector_limits: Dict[str, float] = field(default_factory=lambda: {
        'technology': 0.25,    # Tech capped at 25%
        'broad_market': 0.40,
        'financials': 0.20,
        'healthcare': 0.20,
        'energy': 0.15,
        'default': 0.15,
    })
    
    # Earnings blackout (research: 6x reversal but gap risk)
    earnings_blackout_before: int = 2  # Days before earnings
    earnings_blackout_after: int = 1   # Days after earnings


# =============================================================================
# 3-TIER EXIT CONFIGURATION (Research: Fixes 28% early exit problem)
# =============================================================================

@dataclass
class ScaledExitConfig:
    """3-tier scaled exit parameters"""
    tier_1_pct: float = 0.33           # 33% at 1R
    tier_1_target_r: float = 1.0
    
    tier_2_pct: float = 0.33           # 33% at 2R
    tier_2_target_r: float = 2.0
    
    tier_3_pct: float = 0.34           # 34% trailing
    tier_3_trailing_atr: float = 1.5   # Trail with 1.5 ATR


# =============================================================================
# DAY OF WEEK FILTER (Research: Monday underperforms)
# =============================================================================

@dataclass
class DayOfWeekConfig:
    """Day of week trading preferences"""
    # Research finding: VIX averages +2.16% Monday vs -0.7% Friday
    avoid_entry_days: List[int] = field(default_factory=lambda: [0])  # 0 = Monday
    preferred_entry_days: List[int] = field(default_factory=lambda: [1, 2, 3, 4])  # Tue-Fri
    
    # Reduce position on certain days
    day_multipliers: Dict[int, float] = field(default_factory=lambda: {
        0: 0.5,   # Monday: 50% position
        1: 1.0,   # Tuesday: 100%
        2: 1.0,   # Wednesday: 100%
        3: 1.0,   # Thursday: 100%
        4: 1.0,   # Friday: 100%
    })


# =============================================================================
# TELEGRAM CONFIGURATION
# =============================================================================

@dataclass
class TelegramConfig:
    """Telegram alert settings"""
    enabled: bool = True
    bot_token: Optional[str] = None    # From env: TELEGRAM_BOT_TOKEN
    chat_id: Optional[str] = None      # From env: TELEGRAM_CHAT_ID
    
    # Alert preferences
    send_buy_signals: bool = True
    send_sell_signals: bool = True
    send_tier_exits: bool = True
    send_risk_alerts: bool = True
    send_daily_summary: bool = True
    send_weekly_summary: bool = True
    
    # Quiet hours (no alerts)
    quiet_start_hour: int = 22         # 10 PM
    quiet_end_hour: int = 7            # 7 AM


# =============================================================================
# PAPER TRADING VALIDATION (GO/NO-GO Criteria)
# =============================================================================

@dataclass
class ValidationConfig:
    """Paper trading validation parameters"""
    validation_period_days: int = 14   # 2 weeks
    
    # GO criteria (all must be met)
    min_trades: int = 100
    min_win_rate: float = 0.45         # 45% minimum (target: 68%)
    min_sharpe: float = 1.0
    max_drawdown: float = 0.15         # 15% max
    
    # Warning thresholds
    warn_win_rate: float = 0.55        # Warn if below 55%
    warn_sharpe: float = 1.5           # Warn if below 1.5
    warn_drawdown: float = 0.10        # Warn if above 10%


# =============================================================================
# BACKTEST CONFIGURATION
# =============================================================================

@dataclass  
class BacktestConfig:
    """Backtest and validation parameters"""
    # Data
    default_period: str = "2y"
    
    # CPCV parameters
    cpcv_groups: int = 6
    cpcv_test_groups: int = 2
    
    # Overfitting detection
    max_parameters: int = 5            # Keep parameters low
    max_sharpe_warning: float = 2.5    # Suspicious if > 2.5
    
    # Expected degradation (research: 30-50% normal)
    expected_degradation: float = 0.40  # 40% performance loss backtest→live


# =============================================================================
# MASTER CONFIGURATION
# =============================================================================

@dataclass
class TradingConfig:
    """Master configuration combining all settings"""
    account: AccountConfig = field(default_factory=AccountConfig)
    rsi2: RSI2Config = field(default_factory=RSI2Config)
    dual_momentum: DualMomentumConfig = field(default_factory=DualMomentumConfig)
    crypto_dca: CryptoDCAConfig = field(default_factory=CryptoDCAConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    scaled_exits: ScaledExitConfig = field(default_factory=ScaledExitConfig)
    day_of_week: DayOfWeekConfig = field(default_factory=DayOfWeekConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


# =============================================================================
# LOAD CONFIGURATION
# =============================================================================

def load_config() -> TradingConfig:
    """
    Load configuration with environment variable overrides.
    
    Returns:
        TradingConfig with all settings
    """
    config = TradingConfig()
    
    # Override from environment variables
    if os.getenv('TRADING_PAPER_MODE'):
        config.account.paper_mode = os.getenv('TRADING_PAPER_MODE', 'true').lower() == 'true'
    
    if os.getenv('TRADING_BALANCE'):
        config.account.starting_balance = float(os.getenv('TRADING_BALANCE'))
    
    if os.getenv('TELEGRAM_BOT_TOKEN'):
        config.telegram.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if os.getenv('TELEGRAM_CHAT_ID'):
        config.telegram.chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    return config


# Global config instance
CONFIG = load_config()


# =============================================================================
# MAIN / DISPLAY CONFIG
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TRADING BOT CONFIGURATION")
    print("=" * 60)
    
    config = load_config()
    
    print(f"\n📊 Account:")
    print(f"   Balance: ${config.account.starting_balance:,.0f}")
    print(f"   Paper Mode: {config.account.paper_mode}")
    
    print(f"\n📈 RSI-2 Strategy:")
    print(f"   RSI Oversold: < {config.rsi2.rsi_oversold}")
    print(f"   Consecutive Bars: {config.rsi2.consecutive_bars}+")
    print(f"   ATR Stop: {config.rsi2.atr_stop_multiplier}x")
    
    print(f"\n🔄 Dual Momentum:")
    print(f"   US: {config.dual_momentum.us_equity}")
    print(f"   Intl: {config.dual_momentum.intl_equity}")
    print(f"   Bonds: {config.dual_momentum.bonds}")
    
    print(f"\n⚠️ Risk Management:")
    print(f"   Risk/Trade: {config.risk.base_risk_per_trade:.0%}")
    print(f"   Kelly Fraction: {config.risk.kelly_fraction}")
    print(f"   Max Drawdown: {config.risk.max_drawdown:.0%}")
    
    print(f"\n📊 3-Tier Exits:")
    print(f"   Tier 1: {config.scaled_exits.tier_1_pct:.0%} at {config.scaled_exits.tier_1_target_r}R")
    print(f"   Tier 2: {config.scaled_exits.tier_2_pct:.0%} at {config.scaled_exits.tier_2_target_r}R")
    print(f"   Tier 3: {config.scaled_exits.tier_3_pct:.0%} trailing {config.scaled_exits.tier_3_trailing_atr} ATR")
    
    print(f"\n📅 Day of Week:")
    print(f"   Avoid: Monday (50% position)")
    
    print(f"\n✅ Validation Criteria:")
    print(f"   Min Trades: {config.validation.min_trades}")
    print(f"   Min Win Rate: {config.validation.min_win_rate:.0%}")
    print(f"   Min Sharpe: {config.validation.min_sharpe}")
    print(f"   Max Drawdown: {config.validation.max_drawdown:.0%}")
    
    print("\n" + "=" * 60)
