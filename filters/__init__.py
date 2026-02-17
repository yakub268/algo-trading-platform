"""
Trading Filters Module
======================

Provides signal filters based on research findings to improve trade quality.

Available Filters:
- VolumePriceDivergenceFilter: Detect when volume doesn't confirm price moves
- DoNothingFilter: Skip trading when conditions are unfavorable (high entropy, volatility, etc.)
- RSIDivergenceDetector: Detect hidden divergence patterns (70% win rate documented)
- EntropyFilter: Only trade when market entropy is low (integrated into DoNothingFilter)
- RegimeFilter: Regime-based trade filtering (planned)

Author: Trading Bot Arsenal
Created: January 2026
"""

from .volume_price_divergence import VolumePriceDivergenceFilter
from .do_nothing_filter import DoNothingFilter
from .rsi_divergence import RSIDivergenceDetector
from .ai_filter import AIFilter, get_ai_filter, FilterResult
from .regime_detector import RegimeDetector, MarketRegime, ChangePointDetector

__all__ = [
    'VolumePriceDivergenceFilter',
    'DoNothingFilter',
    'RSIDivergenceDetector',
    'AIFilter',
    'get_ai_filter',
    'FilterResult',
    'RegimeDetector',
    'MarketRegime',
    'ChangePointDetector'
]
