"""
Advanced Options Strategies

Comprehensive collection of options strategies from basic to sophisticated:

Basic Strategies:
- Calls, Puts, Covered Calls, Cash-Secured Puts

Spreads:
- Bull/Bear Call Spreads, Bull/Bear Put Spreads
- Iron Condors, Iron Butterflies
- Calendar Spreads, Diagonal Spreads

Volatility Strategies:
- Straddles, Strangles
- Butterflies, Condors
- Ratio Spreads

Income Strategies:
- Covered Calls, Wheel Strategy
- Credit Spreads, Iron Condors

Advanced Strategies:
- Collar, Protective Put
- Synthetic Positions
- Multi-leg complex strategies
"""

from .basic import CallStrategy, PutStrategy, CoveredCall, CashSecuredPut
from .spreads import (
    BullCallSpread, BearCallSpread, BullPutSpread, BearPutSpread,
    IronCondor, IronButterfly, Calendar, DiagonalSpread
)
from .volatility import Straddle, Strangle, LongButterfly, ShortButterfly, RatioSpread
from .income import WheelStrategy, CoveredCallStrategy, CreditSpreadStrategy
from .advanced import Collar, ProtectivePut, SyntheticCall, SyntheticPut, Conversion, Reversal
from .strategy_analyzer import StrategyAnalyzer, StrategyComparison

__all__ = [
    # Basic strategies
    'CallStrategy',
    'PutStrategy',
    'CoveredCall',
    'CashSecuredPut',

    # Spreads
    'BullCallSpread',
    'BearCallSpread',
    'BullPutSpread',
    'BearPutSpread',
    'IronCondor',
    'IronButterfly',
    'Calendar',
    'DiagonalSpread',

    # Volatility
    'Straddle',
    'Strangle',
    'LongButterfly',
    'ShortButterfly',
    'RatioSpread',

    # Income
    'WheelStrategy',
    'CoveredCallStrategy',
    'CreditSpreadStrategy',

    # Advanced
    'Collar',
    'ProtectivePut',
    'SyntheticCall',
    'SyntheticPut',
    'Conversion',
    'Reversal',

    # Analysis
    'StrategyAnalyzer',
    'StrategyComparison'
]