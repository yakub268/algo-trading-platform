"""
Trading Bots

Platform-specific trading bots:
- AlpacaFOMCTrader: FOMC drift strategy using direct alpaca-py
- KalshiBot: Prediction market trading
- OANDABot: Forex trading
"""

# Lazy imports to avoid dependency issues
__all__ = ['AlpacaFOMCTrader', 'KalshiBot', 'OANDABot']


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == 'AlpacaFOMCTrader':
        from .alpaca_fomc_trader import AlpacaFOMCTrader
        return AlpacaFOMCTrader
    elif name == 'KalshiBot':
        from .kalshi_bot import KalshiBot
        return KalshiBot
    elif name == 'OANDABot':
        from .oanda_bot import OANDABot
        return OANDABot
    raise AttributeError(f"module 'bots' has no attribute '{name}'")
