"""
Aggressive Trading Bots
=======================

High-risk, high-reward strategies for USDC crypto pairs.
These bots require HIGH_RISK_ENABLED=true in config.

Available Bots:
- MomentumScalper: Scans for momentum coins, quick scalps with trailing stops
- BreakoutHunter: Volume spike + price breakout detection on Coinbase USDC pairs
- MultiMomentumBot: Rotates capital into top 3 performing coins every 4 hours
- MemeCoinSniper: Targets volatile meme coins (PEPE, DOGE, SHIB, etc.)
- RSIExtremesBot: Mean reversion at RSI extremes (RSI < 20 buy, RSI > 80 sell)

IMPORTANT: These strategies use larger position sizes and shorter timeframes.
Only enable with capital you can afford to lose.
"""

# Conditional imports - modules may not exist yet
try:
    from .base_aggressive_bot import BaseAggressiveBot, TradeSignal
except ImportError:
    BaseAggressiveBot = None
    TradeSignal = None

try:
    from .momentum_scalper import MomentumScalper
except ImportError:
    MomentumScalper = None

try:
    from .breakout_hunter import BreakoutHunter
except ImportError:
    BreakoutHunter = None

try:
    from .multi_momentum import MultiMomentumBot
except ImportError:
    MultiMomentumBot = None

try:
    from .meme_sniper import MemeCoinSniper
except ImportError:
    MemeCoinSniper = None

try:
    from .rsi_extremes import RSIExtremesBot
except ImportError:
    RSIExtremesBot = None

__all__ = ['BaseAggressiveBot', 'TradeSignal', 'MomentumScalper', 'BreakoutHunter', 'MultiMomentumBot', 'MemeCoinSniper', 'RSIExtremesBot']
