"""
Trading Bot Configuration

Platform-specific configuration modules.
"""

from .kalshi_config import KALSHI_CONFIG
from .oanda_config import OANDA_CONFIG, FOREX_PAIRS

__all__ = ['KALSHI_CONFIG', 'OANDA_CONFIG', 'FOREX_PAIRS']
